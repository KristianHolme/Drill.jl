function _callbacks_continue(callbacks, hook, cache::RLCache)
    for callback in callbacks
        if !hook(callback, cache)
            return false
        end
    end
    return true
end

# Copy of `obs` whose columns for finished envs hold `final_obs`: the observation that
# followed each transition, before any auto-reset.
function _next_observations(obs::AbstractArray, final_obs::AbstractArray, terminateds, truncateds)
    next_obs = copy(obs)
    for j in eachindex(terminateds, truncateds)
        if terminateds[j] || truncateds[j]
            observation_slot(next_obs, j) .= observation_slot(final_obs, j)
        end
    end
    return next_obs
end

"""
    collect_rollout!(buffer::RolloutBuffer, cache, alg, env; callbacks) -> (fps, success)

Fill `buffer` with `buffer.n_steps` steps of all envs, writing each step straight into the
buffer's arrays. `success` is `false` when a callback stopped collection.
"""
function collect_rollout!(
        buffer::RolloutBuffer,
        cache::RLCache,
        alg::OnPolicyAlgorithm,
        env::AbstractParallelEnv;
        callbacks = cache.callbacks,
    )
    t_start = time()
    n_steps = buffer.n_steps
    act_space = action_space(env)
    reset!(buffer)
    obs = observe(env)
    for t in 1:n_steps
        if !_callbacks_continue(callbacks, on_step, cache)
            @warn "Collecting trajectories stopped due to callback failure"
            fps = (t - 1) * buffer.n_envs / max(time() - t_start, eps(Float64))
            return fps, false
        end
        actions, values, logprobs = get_action_and_values(cache, obs)
        env_actions = _env_action.(Ref(cache), actions)
        new_obs, rewards, terminateds, truncateds, final_obs, _ = step!(env, env_actions)
        store_step!(buffer, t, obs, batch(actions, act_space), rewards, logprobs, values, terminateds, truncateds)

        # Bootstrap where an episode is cut short: truncation, or the end of the rollout.
        needs_bootstrap = t == n_steps ? .!terminateds : truncateds
        if any(needs_bootstrap)
            next_obs = _next_observations(new_obs, final_obs, terminateds, truncateds)
            next_values = predict_values(cache, next_obs)
            inds = step_indices(buffer, t)
            for j in findall(needs_bootstrap)
                buffer.bootstrap_values[inds[j]] = next_values[j]
            end
        end
        obs = new_obs
    end
    fps = n_steps * buffer.n_envs / max(time() - t_start, eps(Float64))
    return fps, true
end

function prepare_rollout!(buffer::RolloutBuffer, alg::PPO)
    compute_gae!(buffer, alg.gamma, alg.gae_lambda)
    return buffer
end

"""
    collect_rollout!(buffer::ReplayBuffer, cache, alg, env, n_steps, progress_meter = nothing;
                     callbacks, use_random_actions = false) -> (fps, success)

Step all envs `n_steps` times and add every transition to `buffer`. With
`use_random_actions`, actions are sampled uniformly from the action space and stored in
policy space.
"""
function collect_rollout!(
        buffer::ReplayBuffer,
        cache::RLCache,
        alg::OffPolicyAlgorithm,
        env::AbstractParallelEnv,
        n_steps::Int,
        progress_meter::Union{Progress, Nothing} = nothing;
        callbacks = cache.callbacks,
        use_random_actions::Bool = false,
    )
    t_start = time()
    act_space = action_space(env)
    n_envs = number_of_envs(env)
    obs = observe(env)
    for t in 1:n_steps
        if !_callbacks_continue(callbacks, on_step, cache)
            @warn "Collecting trajectories stopped due to callback failure"
            fps = (t - 1) * n_envs / max(time() - t_start, eps(Float64))
            return fps, false
        end
        if use_random_actions
            # Sample in env space; store the policy-space equivalent for training.
            env_actions = rand(cache.rng, act_space, n_envs)
            actions = from_env.(Ref(cache.adapter), env_actions, Ref(act_space))
        else
            actions = predict_actions(cache, obs; raw = true)
            env_actions = _env_action.(Ref(cache), actions)
        end
        new_obs, rewards, terminateds, truncateds, final_obs, _ = step!(env, env_actions)
        next_obs = _next_observations(new_obs, final_obs, terminateds, truncateds)
        add_transitions!(buffer, obs, batch(actions, act_space), rewards, terminateds, truncateds, next_obs)
        obs = new_obs
        !isnothing(progress_meter) && next!(progress_meter, step = n_envs)
    end
    fps = n_steps * n_envs / max(time() - t_start, eps(Float64))
    return fps, true
end
