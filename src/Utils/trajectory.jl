# Trajectory utility functions

function collect_trajectory(
        predict_actions,
        rng::AbstractRNG,
        env;
        max_steps::Union{Int, Nothing} = nothing,
        norm_env = nothing,
        deterministic::Bool = true,
    )
    reset!(env)
    original_training = is_training(env)
    env = set_training(env, false)
    observations = []
    actions = []
    rewards = []
    while !(terminated(env) || truncated(env))
        observation = observe(env)
        obs_to_agent = copy(observation)
        if env isa ScalingWrapperEnv
            unscale_observation!(observation, env)
        end
        original_observation = observation
        push!(observations, original_observation)
        if norm_env !== nothing
            normalize_obs!(obs_to_agent, norm_env)
        end

        agent_actions = predict_actions([obs_to_agent]; deterministic, rng)
        agent_action = first(agent_actions)
        if env isa ScalingWrapperEnv
            unscale_action!(agent_action, env)
        end
        env_action = agent_action
        push!(actions, env_action)
        reward = act!(env, agent_action)

        push!(rewards, reward)
        if max_steps !== nothing && length(observations) >= max_steps
            @warn "Max steps reached"
            break
        end
    end
    final_observation = observe(env)
    push!(observations, final_observation)
    # FIXME: this doesn't really work as expected, the change here is not affecting the real env
    env = set_training(env, original_training)
    return observations, actions, rewards
end
