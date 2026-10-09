using MLUtils: DataLoader
using SciMLBase: ReturnCode
using Statistics: mean, var
using TimerOutputs: @timeit

import DrillInterface: action_space, number_of_envs

import ..Solve: RLCache, collect_rollout!, prepare_rollout!, add_steps!, add_gradient_update!,
    _record_stat!, _mark_complete!, _callbacks_continue, update_training_progress!,
    current_device, gradient_backend, host_metrics, requires_fixed_shapes, run_update
import ..DrillLogging: increment_step!, log_scalar!, log_stats
const _Drill = parentmodule(@__MODULE__)
const on_rollout_start = _Drill.on_rollout_start
const on_rollout_end = _Drill.on_rollout_end

function train_step!(cache::RLCache{<:Any, <:PPO}, alg::PPO)
    env = cache.prob.env
    n_steps = alg.n_steps
    n_envs = number_of_envs(env)
    _record_stat!(cache, :learning_rates, alg.learning_rate)

    if !_callbacks_continue(cache.callbacks, on_rollout_start, cache)
        cache.retcode = ReturnCode.Terminated
        return cache
    end

    fps, success = @timeit cache.timer "collect_rollout" collect_rollout!(
        cache.buffer,
        cache,
        alg,
        env;
        callbacks = cache.callbacks,
    )
    if !success
        cache.retcode = ReturnCode.Terminated
        return cache
    end
    prepare_rollout!(cache.buffer, alg)
    add_steps!(cache, n_steps * n_envs)
    increment_step!(cache.logger, n_steps * n_envs)
    log_scalar!(cache.logger, "env/fps", fps)
    log_stats(env, cache.logger)
    _record_stat!(cache, :fps, Float32(fps))

    if !_callbacks_continue(cache.callbacks, on_rollout_end, cache)
        cache.retcode = ReturnCode.Terminated
        return cache
    end

    metrics = @timeit cache.timer "update" _ppo_epochs!(cache, alg)

    explained_variance = 1 - var(cache.buffer.values .- cache.buffer.returns) / var(cache.buffer.returns)
    means = map(mean, metrics)
    _record_stat!(cache, :entropy_losses, means.entropy_loss)
    _record_stat!(cache, :policy_losses, means.policy_loss)
    _record_stat!(cache, :value_losses, means.value_loss)
    _record_stat!(cache, :approx_kl_divs, means.approx_kl_div)
    _record_stat!(cache, :clip_fractions, means.clip_fraction)
    _record_stat!(cache, :losses, means.loss)
    _record_stat!(cache, :grad_norms, means.grad_norm)
    _record_stat!(cache, :explained_variances, Float32(explained_variance))

    log_scalar!(cache.logger, "train/entropy_loss", means.entropy_loss)
    log_scalar!(cache.logger, "train/explained_variance", explained_variance)
    log_scalar!(cache.logger, "train/policy_loss", means.policy_loss)
    log_scalar!(cache.logger, "train/value_loss", means.value_loss)
    log_scalar!(cache.logger, "train/approx_kl_div", means.approx_kl_div)
    log_scalar!(cache.logger, "train/clip_fraction", means.clip_fraction)
    log_scalar!(cache.logger, "train/loss", means.loss)
    log_scalar!(cache.logger, "train/grad_norm", means.grad_norm)
    log_scalar!(cache.logger, "train/learning_rate", alg.learning_rate)
    ps = parameters(cache)
    if ps isa NamedTuple && haskey(ps, :log_std)
        log_scalar!(cache.logger, "train/std", mean(exp.(Array(ps.log_std))))
    end

    update_training_progress!(
        cache,
        n_steps * n_envs;
        showvalues = [
            ("explained_variance", explained_variance),
            ("entropy_loss", means.entropy_loss),
            ("policy_loss", means.policy_loss),
            ("value_loss", means.value_loss),
            ("approx_kl_div", means.approx_kl_div),
            ("clip_fraction", means.clip_fraction),
            ("loss", means.loss),
            ("fps", fps),
            ("grad_norm", means.grad_norm),
            ("learning_rate", alg.learning_rate),
        ],
    )
    return _mark_complete!(cache)
end

# The PPO epoch loop: one pure `ppo_update` per minibatch. Returns the per-minibatch
# metrics as host vectors.
function _ppo_epochs!(cache::RLCache, alg::PPO)
    buffer = cache.buffer
    train_actions = prepare_training_actions(buffer.actions, action_space(buffer))
    dev = current_device(parameters(cache))
    data_loader = DataLoader(
        (buffer.observations, train_actions, buffer.advantages, buffer.returns, buffer.logprobs, buffer.values);
        batchsize = alg.batch_size,
        shuffle = true,
        partial = !requires_fixed_shapes(dev),
        rng = cache.rng,
    )
    ad = gradient_backend(dev, cache.ad_type)
    kl_threshold = target_kl(alg)
    names = (:loss, :policy_loss, :value_loss, :entropy_loss, :approx_kl_div, :clip_fraction, :grad_norm)
    history = NamedTuple{names}(ntuple(_ -> Float32[], length(names)))
    learner = cache.learner
    for epoch in 1:alg.epochs
        stop = false
        for batch in dev(data_loader)
            # The update may reuse the learner's memory; keep a copy to fall back to when
            # the KL early stop rejects the step.
            previous = isnothing(kl_threshold) ? nothing : deepcopy(learner)
            learner, metrics = run_update(dev, cache, ppo_update, alg, cache.model, learner, batch, ad)
            metrics = host_metrics(metrics)
            isfinite(metrics.loss) || error("PPO loss is not finite (epoch $epoch): $(metrics.loss)")
            if !isnothing(kl_threshold) && metrics.approx_kl_div > 1.5f0 * kl_threshold
                learner = previous
                stop = true
                break
            end
            add_gradient_update!(cache)
            foreach(n -> push!(history[n], metrics[n]), names)
        end
        stop && break
    end
    cache.learner = learner
    return history
end
