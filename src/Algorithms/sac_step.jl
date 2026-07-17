using Lux: Training
using Optimisers: adjust!
using SciMLBase: ReturnCode
using Statistics: mean
using TimerOutputs: @timeit

import DrillInterface: action_space, number_of_envs

import ..Solve: RLCache, collect_rollout!, add_steps!, add_gradient_update!,
    _record_stat!, _mark_complete!, _callbacks_continue, get_device
import ..DrillLogging: increment_step!, log_scalar!, log_stats, set_step!
import ..Buffers: get_data_loader
import ..Utils: nested_norm, polyak_update!
const _Drill = parentmodule(@__MODULE__)
const on_rollout_start = _Drill.on_rollout_start
const on_rollout_end = _Drill.on_rollout_end

function _sac_update!(cache::RLCache{<:Any, <:SAC}, batch_data)
    alg = cache.alg
    model = cache.model
    ts = cache.train_state
    full_ps = parameters(ts)
    full_st = states(ts)
    ent_loss = nothing
    if alg.ent_coef isa AutoEntropyCoefficient
        target_entropy = get_target_entropy(alg.ent_coef, action_space(cache.prob.env))
        _, log_probs_pi, _ = action_log_prob(
            model,
            batch_data.observations,
            full_ps,
            full_st;
            rng = cache.rng,
        )
        c = mean(log_probs_pi .+ target_entropy)
        ent_grad, ent_loss_val, _, ent_ts = Training.compute_gradients(
            cache.ad_type,
            SACEntropyObjective(),
            (; c),
            ts.ent_ts,
        )
        ts.ent_ts = Training.apply_gradients!(ent_ts, ent_grad)
        ent_loss = ent_loss_val
    end
    target_q_values = compute_target_q_values(
        alg,
        model,
        full_ps,
        full_st,
        (
            next_observations = batch_data.next_observations,
            terminated = batch_data.terminated,
            log_ent_coef = entropy_parameters(ts),
            rewards = batch_data.rewards,
            target_ps = ts.target_parameters,
            target_st = ts.target_states,
        );
        rng = cache.rng,
    )
    critic_data = (
        observations = batch_data.observations,
        actions = batch_data.actions,
        target_q_values = target_q_values,
        actor_ps = ts.actor_ts.parameters,
        actor_st = ts.actor_ts.states,
    )
    critic_grad, critic_loss, critic_stats, critic_ts = Training.compute_gradients(
        cache.ad_type,
        SACCriticObjective(alg, cache.rng),
        critic_data,
        ts.critic_ts,
    )
    ts.critic_ts = Training.apply_gradients!(critic_ts, critic_grad)

    ent_coef = Float32(entropy_coefficient(ts))
    actor_data = (
        observations = batch_data.observations,
        ent_coef = ent_coef,
        critic_ps = ts.critic_ts.parameters,
        critic_st = ts.critic_ts.states,
    )
    actor_grad, actor_loss, _, actor_ts = Training.compute_gradients(
        cache.ad_type,
        SACActorObjective(alg, cache.rng),
        actor_data,
        ts.actor_ts,
    )
    ts.actor_ts = Training.apply_gradients!(actor_ts, actor_grad)

    add_gradient_update!(cache)
    if cache.gradient_updates % alg.target_update_interval == 0
        ts.target_states = deepcopy(ts.critic_ts.states)
        polyak_update!(ts.target_parameters, ts.critic_ts.parameters, alg.tau)
    end
    T = typeof(alg.learning_rate)
    total_grad_norm = sqrt(nested_norm(critic_grad, T)^2 + nested_norm(actor_grad, T)^2)
    return (
        actor_loss = actor_loss,
        critic_loss = critic_loss,
        entropy_loss = ent_loss,
        mean_q_values = critic_stats.mean_q_values,
        entropy_coefficient = entropy_coefficient(ts),
        grad_norm = total_grad_norm,
    )
end

function train_step!(cache::RLCache{<:Any, <:SAC}, alg::SAC)
    env = cache.prob.env
    n_envs = number_of_envs(env)
    cache.workspace[:sac_iteration] = get(cache.workspace, :sac_iteration, 0) + 1
    n_steps = get(cache.workspace, :next_collect_steps, alg.train_freq)
    use_random_actions = cache.workspace[:sac_iteration] == 1 && alg.start_steps > 0

    adjust!(cache.train_state, alg.learning_rate)
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
        data_loader = get_data_loader(cache.buffer, alg.batch_size, n_updates, true, true, cache.rng)
        dev = get_device(parameters(cache))
        data_iter = dev !== nothing ? dev(data_loader) : data_loader
        for batch_data in data_iter
            stats_step = _sac_update!(cache, batch_data)
            stats_step.entropy_loss !== nothing &&
                _record_stat!(cache, :entropy_losses, Float32(stats_step.entropy_loss))
            _record_stat!(cache, :critic_losses, Float32(stats_step.critic_loss))
            _record_stat!(cache, :actor_losses, Float32(stats_step.actor_loss))
            _record_stat!(cache, :q_values, Float32(stats_step.mean_q_values))
            _record_stat!(cache, :entropy_coefficients, Float32(stats_step.entropy_coefficient))
            _record_stat!(cache, :learning_rates, alg.learning_rate)
            _record_stat!(cache, :grad_norms, Float32(stats_step.grad_norm))
        end
    end
    set_step!(cache.logger, cache.steps_taken)
    log_scalar!(cache.logger, "train/total_steps", cache.steps_taken)
    return _mark_complete!(cache)
end
