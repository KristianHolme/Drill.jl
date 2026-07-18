function _callbacks_continue(callbacks, hook, cache::RLCache)
    for callback in callbacks
        if !hook(callback, cache)
            return false
        end
    end
    return true
end

function collect_trajectories(
        cache::RLCache,
        env::AbstractParallelEnv,
        alg::OnPolicyAlgorithm,
        n_steps::Int;
        callbacks = cache.callbacks,
    )
    trajectories = Trajectory[]
    obs_space = observation_space(env)
    act_space = action_space(env)
    n_envs = number_of_envs(env)
    current_trajectories = [Trajectory(obs_space, act_space) for _ in 1:n_envs]
    new_obs = observe(env)
    for i in 1:n_steps
        if !_callbacks_continue(callbacks, on_step, cache)
            @warn "Collecting trajectories stopped due to callback failure"
            return trajectories, false
        end
        observations = new_obs
        actions, values, logprobs = get_action_and_values(cache, observations)
        processed_actions = _env_action.(Ref(cache), actions)
        rewards, terminateds, truncateds, infos = act!(env, processed_actions)
        new_obs = observe(env)
        for j in 1:n_envs
            push!(current_trajectories[j].observations, observations[j])
            push!(current_trajectories[j].actions, actions[j])
            push!(current_trajectories[j].rewards, rewards[j])
            push!(current_trajectories[j].logprobs, logprobs[j])
            push!(current_trajectories[j].values, values[j])
            if terminateds[j] || truncateds[j] || i == n_steps
                current_trajectories[j].terminated = terminateds[j]
                current_trajectories[j].truncated = truncateds[j]
                if truncateds[j] && haskey(infos[j], "terminal_observation")
                    last_observation = infos[j]["terminal_observation"]
                    current_trajectories[j].bootstrap_value = predict_values(cache, [last_observation])[1]
                elseif !terminateds[j] && !truncateds[j] && i == n_steps
                    current_trajectories[j].bootstrap_value = predict_values(cache, [new_obs[j]])[1]
                end
                push!(trajectories, current_trajectories[j])
                current_trajectories[j] = Trajectory(obs_space, act_space)
            end
        end
    end
    return trajectories, true
end

function collect_rollout!(
        rollout_buffer::RolloutBuffer,
        cache::RLCache,
        alg::OnPolicyAlgorithm,
        env::AbstractParallelEnv;
        callbacks = cache.callbacks,
    )
    t_start = time()
    trajectories, success = collect_trajectories(cache, env, alg, rollout_buffer.n_steps; callbacks)
    t_collect = time() - t_start
    total_steps = sum(length.(trajectories))
    fps = total_steps / max(t_collect, eps(Float64))
    if !success
        return fps, false
    end
    pack_trajectories!(rollout_buffer, trajectories)
    return fps, true
end

function prepare_rollout!(buffer::RolloutBuffer, alg::PPO)
    compute_gae!(buffer, alg.gamma, alg.gae_lambda)
    return buffer
end

function collect_trajectories(
        cache::RLCache,
        alg::OffPolicyAlgorithm,
        env::AbstractParallelEnv,
        n_steps::Int,
        progress_meter::Union{Progress, Nothing} = nothing;
        callbacks = cache.callbacks,
        use_random_actions::Bool = false,
    )
    trajectories = OffPolicyTrajectory[]
    obs_space = observation_space(env)
    act_space = action_space(env)
    n_envs = number_of_envs(env)
    current_trajectories = [OffPolicyTrajectory(obs_space, act_space) for _ in 1:n_envs]
    new_obs = observe(env)
    for i in 1:n_steps
        if !_callbacks_continue(callbacks, on_step, cache)
            @warn "Collecting trajectories stopped due to callback failure"
            return trajectories, false
        end
        observations = new_obs
        if use_random_actions
            actions = rand(cache.rng, act_space, length(observations))
            processed_actions = actions
        else
            actions = predict_actions(cache, observations; raw = true)
            processed_actions = _env_action.(Ref(cache), actions)
        end
        rewards, terminateds, truncateds, infos = act!(env, processed_actions)
        new_obs = observe(env)
        for j in 1:n_envs
            push!(current_trajectories[j].observations, observations[j])
            push!(current_trajectories[j].actions, actions[j])
            push!(current_trajectories[j].rewards, rewards[j])
            if terminateds[j] || truncateds[j] || i == n_steps
                current_trajectories[j].terminated = terminateds[j]
                current_trajectories[j].truncated = truncateds[j]
                if truncateds[j] && haskey(infos[j], "terminal_observation")
                    current_trajectories[j].truncated_observation = infos[j]["terminal_observation"]
                elseif !terminateds[j] && !truncateds[j] && i == n_steps
                    current_trajectories[j].truncated_observation = new_obs[j]
                end
                push!(trajectories, current_trajectories[j])
                current_trajectories[j] = OffPolicyTrajectory(obs_space, act_space)
            end
        end
        !isnothing(progress_meter) && next!(progress_meter, step = number_of_envs(env))
    end
    return trajectories, true
end

function collect_rollout!(
        buffer::ReplayBuffer,
        cache::RLCache,
        alg::OffPolicyAlgorithm,
        env::AbstractParallelEnv,
        n_steps::Int,
        progress_meter::Union{Progress, Nothing} = nothing;
        kwargs...,
    )
    t_start = time()
    trajectories, success = collect_trajectories(cache, alg, env, n_steps, progress_meter; kwargs...)
    t_collect = time() - t_start
    total_steps = sum(length.(trajectories))
    fps = total_steps / max(t_collect, eps(Float64))
    if !success
        return fps, false
    end
    for traj in trajectories
        push!(buffer, traj)
    end
    return fps, true
end
