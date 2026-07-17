function _record_stat!(cache::RLCache, key::Symbol, value)
    values = get!(cache.stats, key) do
        typeof(value)[]
    end
    push!(values, value)
    return values
end

function _mark_complete!(cache::RLCache)
    if cache.steps_taken >= cache.max_steps && cache.retcode != SciMLBase.ReturnCode.Terminated
        cache.retcode = SciMLBase.ReturnCode.Success
    end
    return cache
end

function CommonSolve.step!(cache::RLCache{<:Any, <:PPO})
    alg = cache.alg
    env = cache.prob.env
    n_steps = alg.n_steps
    n_envs = number_of_envs(env)
    Optimisers.adjust!(cache.train_state, alg.learning_rate)
    _record_stat!(cache, :learning_rates, alg.learning_rate)

    if !_callbacks_continue(cache.callbacks, on_rollout_start, cache)
        cache.retcode = SciMLBase.ReturnCode.Terminated
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
        cache.retcode = SciMLBase.ReturnCode.Terminated
        return cache
    end
    prepare_rollout!(cache.buffer, alg)
    add_steps!(cache, n_steps * n_envs)
    increment_step!(cache.logger, n_steps * n_envs)
    log_scalar!(cache.logger, "env/fps", fps)
    log_stats(env, cache.logger)
    _record_stat!(cache, :fps, Float32(fps))

    if !_callbacks_continue(cache.callbacks, on_rollout_end, cache)
        cache.retcode = SciMLBase.ReturnCode.Terminated
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
    dev = MLDataDevices.get_device(parameters(cache))
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
            grads, loss_val, stats, train_state = Lux.Training.compute_gradients(
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
            train_state = Lux.Training.apply_gradients!(train_state, grads)
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
        ent_grad, ent_loss_val, _, ent_ts = Lux.Training.compute_gradients(
            cache.ad_type,
            SACEntropyObjective(),
            (; c),
            ts.ent_ts,
        )
        ts.ent_ts = Lux.Training.apply_gradients!(ent_ts, ent_grad)
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
    critic_grad, critic_loss, critic_stats, critic_ts = Lux.Training.compute_gradients(
        cache.ad_type,
        SACCriticObjective(alg, cache.rng),
        critic_data,
        ts.critic_ts,
    )
    ts.critic_ts = Lux.Training.apply_gradients!(critic_ts, critic_grad)

    ent_coef = Float32(entropy_coefficient(ts))
    actor_data = (
        observations = batch_data.observations,
        ent_coef = ent_coef,
        critic_ps = ts.critic_ts.parameters,
        critic_st = ts.critic_ts.states,
    )
    actor_grad, actor_loss, _, actor_ts = Lux.Training.compute_gradients(
        cache.ad_type,
        SACActorObjective(alg, cache.rng),
        actor_data,
        ts.actor_ts,
    )
    ts.actor_ts = Lux.Training.apply_gradients!(actor_ts, actor_grad)

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

function CommonSolve.step!(cache::RLCache{<:Any, <:SAC})
    alg = cache.alg
    env = cache.prob.env
    n_envs = number_of_envs(env)
    cache.workspace[:sac_iteration] = get(cache.workspace, :sac_iteration, 0) + 1
    n_steps = get(cache.workspace, :next_collect_steps, alg.train_freq)
    use_random_actions = cache.workspace[:sac_iteration] == 1 && alg.start_steps > 0

    Optimisers.adjust!(cache.train_state, alg.learning_rate)
    if !_callbacks_continue(cache.callbacks, on_rollout_start, cache)
        cache.retcode = SciMLBase.ReturnCode.Terminated
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
        cache.retcode = SciMLBase.ReturnCode.Terminated
        return cache
    end
    add_steps!(cache, n_steps * n_envs)
    increment_step!(cache.logger, n_steps * n_envs)
    cache.workspace[:next_collect_steps] = alg.train_freq
    _record_stat!(cache, :fps, Float32(fps))
    log_scalar!(cache.logger, "env/fps", fps)
    log_stats(env, cache.logger)

    if !_callbacks_continue(cache.callbacks, on_rollout_end, cache)
        cache.retcode = SciMLBase.ReturnCode.Terminated
        return cache
    end

    n_updates = get_gradient_steps(alg, alg.train_freq, n_envs)
    if length(cache.buffer) > 0 && n_updates > 0
        data_loader = get_data_loader(cache.buffer, alg.batch_size, n_updates, true, true, cache.rng)
        dev = MLDataDevices.get_device(parameters(cache))
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
