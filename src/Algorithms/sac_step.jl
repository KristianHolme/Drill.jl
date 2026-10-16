using SciMLBase: ReturnCode
using Statistics: mean
using TimerOutputs: @timeit

import DrillInterface: action_space, number_of_envs

import ..Solve: RLCache, collect_rollout!, add_steps!, add_gradient_update!,
    _record_stat!, _mark_complete!, _callbacks_continue, update_training_progress!,
    latest_stat, current_device, gradient_backend, host_metrics, run_update
import ..DrillLogging: increment_step!, log_scalar!, log_stats, set_step!
import ..Buffers: get_data_loader
const _Drill = parentmodule(@__MODULE__)
const on_rollout_start = _Drill.on_rollout_start
const on_rollout_end = _Drill.on_rollout_end

function train_step!(cache::RLCache{<:Any, <:SAC}, alg::SAC)
    env = cache.prob.env
    n_envs = number_of_envs(env)
    cache.workspace[:sac_iteration] = get(cache.workspace, :sac_iteration, 0) + 1
    n_steps = get(cache.workspace, :next_collect_steps, alg.train_freq)
    use_random_actions = cache.workspace[:sac_iteration] == 1 && alg.start_steps > 0

    if !_callbacks_continue(cache.callbacks, on_rollout_start, cache)
        cache.retcode = ReturnCode.Terminated
        return cache
    end
    fps, success = @timeit cache.timer "collect_rollout" collect_rollout!(
        cache.buffer,
        cache,
        alg,
        env,
        n_steps;
        callbacks = cache.callbacks,
        use_random_actions,
    )
    if !success
        cache.retcode = ReturnCode.Terminated
        return cache
    end
    add_steps!(cache, n_steps * n_envs)
    increment_step!(cache.logger, n_steps * n_envs)
    cache.workspace[:next_collect_steps] = alg.train_freq
    _record_stat!(cache, :fps, Float32(fps))
    log_scalar!(cache.logger, "env/fps", fps)
    log_stats(env, cache.logger)

    if !_callbacks_continue(cache.callbacks, on_rollout_end, cache)
        cache.retcode = ReturnCode.Terminated
        return cache
    end

    n_updates = get_gradient_steps(alg, alg.train_freq, n_envs)
    if length(cache.buffer) > 0 && n_updates > 0
        @timeit cache.timer "update" _sac_updates!(cache, alg, n_updates)
    end
    set_step!(cache.logger, cache.steps_taken)
    log_scalar!(cache.logger, "train/total_steps", cache.steps_taken)

    showvalues = [
        ("fps", fps),
        ("actor_loss", latest_stat(cache, :actor_losses)),
        ("critic_loss", latest_stat(cache, :critic_losses)),
        ("entropy_loss", latest_stat(cache, :entropy_losses)),
        ("mean_q_values", latest_stat(cache, :q_values)),
        ("entropy_coefficient", latest_stat(cache, :entropy_coefficients)),
        ("grad_norm", latest_stat(cache, :grad_norms)),
        ("learning_rate", alg.learning_rate),
    ]
    filter!(pair -> last(pair) !== nothing, showvalues)
    update_training_progress!(cache, n_steps * n_envs; showvalues)
    return _mark_complete!(cache)
end

function _sac_updates!(cache::RLCache, alg::SAC, n_updates::Int)
    data_loader = get_data_loader(cache.buffer, alg.batch_size, n_updates, true, false, cache.rng)
    dev = current_device(parameters(cache))
    ad = gradient_backend(dev, cache.ad_type)
    target_entropy = get_target_entropy(alg.ent_coef, action_space(cache.buffer))
    learner = cache.learner
    for batch in dev(data_loader)
        update_target = Val((cache.gradient_updates + 1) % alg.target_update_interval == 0)
        learner, metrics = run_update(dev, cache, sac_update, alg, cache.model, learner, batch, ad, target_entropy, update_target)
        add_gradient_update!(cache)
        metrics = host_metrics(metrics)
        isfinite(metrics.critic_loss) || error("SAC critic loss is not finite: $(metrics.critic_loss)")
        if alg.ent_coef isa AutoEntropyCoefficient
            _record_stat!(cache, :entropy_losses, metrics.entropy_loss)
        end
        _record_stat!(cache, :critic_losses, metrics.critic_loss)
        _record_stat!(cache, :actor_losses, metrics.actor_loss)
        _record_stat!(cache, :q_values, metrics.mean_q_values)
        _record_stat!(cache, :entropy_coefficients, metrics.entropy_coefficient)
        _record_stat!(cache, :learning_rates, alg.learning_rate)
        _record_stat!(cache, :grad_norms, metrics.grad_norm)
    end
    cache.learner = learner
    return nothing
end
