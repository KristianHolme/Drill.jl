using Lux: Training
using MLUtils: DataLoader
using Optimisers: adjust!
using SciMLBase: ReturnCode
using Statistics: mean, var
using TimerOutputs: @timeit

import DrillInterface: action_space, number_of_envs

import ..Solve: RLCache, collect_rollout!, prepare_rollout!, add_steps!, add_gradient_update!,
    get_device, _record_stat!, _mark_complete!, _callbacks_continue
import ..DrillLogging: increment_step!, log_scalar!, log_stats
import ..Utils: nested_has_inf, nested_has_nan, nested_norm, nested_scale!
const _Drill = parentmodule(@__MODULE__)
const on_rollout_start = _Drill.on_rollout_start
const on_rollout_end = _Drill.on_rollout_end

function train_step!(cache::RLCache{<:Any, <:PPO}, alg::PPO)
    env = cache.prob.env
    n_steps = alg.n_steps
    n_envs = number_of_envs(env)
    adjust!(cache.train_state, alg.learning_rate)
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

    train_actions = prepare_training_actions(cache.buffer.actions, action_space(env))
    data_loader = DataLoader(
        (
            cache.buffer.observations,
            train_actions,
            cache.buffer.advantages,
            cache.buffer.returns,
            cache.buffer.logprobs,
            cache.buffer.values,
        );
        batchsize = alg.batch_size,
        shuffle = true,
        parallel = true,
        rng = cache.rng,
    )
    dev = get_device(parameters(cache))
    entropy_losses = Float32[]
    policy_losses = Float32[]
    value_losses = Float32[]
    losses = Float32[]
    approx_kl_divs = Float32[]
    clip_fractions = Float32[]
    grad_norms = Float32[]
    train_state = lux_train_state(cache.train_state)
    T = typeof(alg.learning_rate)
    continue_training = true
    for epoch in 1:alg.epochs
        data_iter = dev !== nothing ? dev(data_loader) : data_loader
        for (i_batch, batch_data) in enumerate(data_iter)
            batch_data = maybe_normalize_batch_data(batch_data, alg.advantage_strategy)
            grads, loss_val, stats, train_state = Training.compute_gradients(
                cache.ad_type,
                alg,
                batch_data,
                train_state,
            )
            @assert !nested_has_nan(grads) "gradient contains nan, epoch $epoch, batch $i_batch"
            @assert !nested_has_inf(grads) "gradient not finite, epoch $epoch, batch $i_batch"
            current_grad_norm = nested_norm(grads, T)
            if current_grad_norm > alg.max_grad_norm
                nested_scale!(grads, alg.max_grad_norm, current_grad_norm)
            end
            kl_threshold = target_kl(alg)
            if !isnothing(kl_threshold) && stats.approx_kl_div > T(1.5) * kl_threshold
                continue_training = false
                break
            end
            train_state = Training.apply_gradients!(train_state, grads)
            set_lux_train_state!(cache.train_state, train_state)
            add_gradient_update!(cache)
            push!(entropy_losses, Float32(stats.entropy_loss))
            push!(policy_losses, Float32(stats.policy_loss))
            push!(value_losses, Float32(stats.value_loss))
            push!(approx_kl_divs, Float32(stats.approx_kl_div))
            push!(clip_fractions, Float32(stats.clip_fraction))
            push!(losses, Float32(loss_val))
            push!(grad_norms, Float32(current_grad_norm))
        end
        continue_training || break
    end
    set_lux_train_state!(cache.train_state, train_state)

    explained_variance = 1 - var(cache.buffer.values .- cache.buffer.returns) / var(cache.buffer.returns)
    _record_stat!(cache, :entropy_losses, mean(entropy_losses))
    _record_stat!(cache, :policy_losses, mean(policy_losses))
    _record_stat!(cache, :value_losses, mean(value_losses))
    _record_stat!(cache, :approx_kl_divs, mean(approx_kl_divs))
    _record_stat!(cache, :clip_fractions, mean(clip_fractions))
    _record_stat!(cache, :losses, mean(losses))
    _record_stat!(cache, :grad_norms, mean(grad_norms))
    _record_stat!(cache, :explained_variances, Float32(explained_variance))

    log_scalar!(cache.logger, "train/entropy_loss", mean(entropy_losses))
    log_scalar!(cache.logger, "train/explained_variance", explained_variance)
    log_scalar!(cache.logger, "train/policy_loss", mean(policy_losses))
    log_scalar!(cache.logger, "train/value_loss", mean(value_losses))
    log_scalar!(cache.logger, "train/approx_kl_div", mean(approx_kl_divs))
    log_scalar!(cache.logger, "train/clip_fraction", mean(clip_fractions))
    log_scalar!(cache.logger, "train/loss", mean(losses))
    log_scalar!(cache.logger, "train/grad_norm", mean(grad_norms))
    log_scalar!(cache.logger, "train/learning_rate", alg.learning_rate)
    ps = parameters(cache)
    if ps isa NamedTuple && haskey(ps, :log_std)
        log_scalar!(cache.logger, "train/std", mean(exp.(ps.log_std)))
    end
    return _mark_complete!(cache)
end
