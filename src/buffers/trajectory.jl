# Trajectory-specific functions

function Trajectory{T}(observation_space::AbstractSpace, action_space::AbstractSpace) where {T <: AbstractFloat}
    obs_type = typeof(rand(observation_space))
    action_type = typeof(rand(action_space))
    observations = Array{obs_type}[]
    actions = Array{action_type}[]
    rewards = T[]
    logprobs = T[]
    values = T[]
    terminated = false
    truncated = false
    bootstrap_value = nothing
    return Trajectory{T, obs_type, action_type}(observations, actions, rewards, logprobs, values, terminated, truncated, bootstrap_value)
end

Trajectory(observation_space::AbstractSpace, action_space::AbstractSpace) = Trajectory{Float32}(observation_space, action_space)

Base.length(trajectory::Trajectory) = length(trajectory.rewards)
total_reward(trajectory::Trajectory) = sum(trajectory.rewards)

function collect_trajectories(
        agent::Agent, env::AbstractParallelEnv, alg::AbstractAlgorithm, n_steps::Int;
        callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing
    )
    # reset!(env)
    trajectories = Trajectory[]
    obs_space = observation_space(env)
    act_space = action_space(env)
    n_envs = number_of_envs(env)
    current_trajectories = [Trajectory(obs_space, act_space) for _ in 1:n_envs]
    new_obs = observe(env)
    for i in 1:n_steps
        if !isnothing(callbacks)
            if !all(c -> on_step(c, Base.@locals), callbacks)
                @warn "Collecting trajectories stopped due to callback failure"
                return trajectories, false
            end
        end
        observations = new_obs
        actions, values, logprobs = get_action_and_values(agent, observations)
        adapter = agent.action_adapter
        processed_actions = to_env.(Ref(adapter), actions, Ref(act_space))
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

                # Handle bootstrapping for truncated episodes
                if truncateds[j] && haskey(infos[j], "terminal_observation")
                    last_observation = infos[j]["terminal_observation"]
                    terminal_value = predict_values(agent, [last_observation])[1]
                    current_trajectories[j].bootstrap_value = terminal_value
                end

                # Handle bootstrapping for rollout-limited trajectories (neither terminated nor truncated)
                # We need to bootstrap with the value of the current observation
                if !terminateds[j] && !truncateds[j] && i == n_steps
                    # Get the next observation after last step (which is the current state)
                    #TODO: do this batched for efficiency? maybe remove the 1:n_envs for loop somehow?
                    next_obs = new_obs[j]
                    next_value = predict_values(agent, [next_obs])[1]
                    current_trajectories[j].bootstrap_value = next_value
                end

                push!(trajectories, current_trajectories[j])
                current_trajectories[j] = Trajectory(obs_space, act_space)
            end
        end
    end
    return trajectories, true
end

function compute_advantages!(advantages::AbstractArray, traj::Trajectory, gamma::AbstractFloat, gae_lambda::AbstractFloat)
    n = length(traj.rewards)

    # For terminated episodes, no bootstrapping (bootstrap_value should be nothing)
    # For truncated episodes or rollout-limited trajectories, bootstrap with the next state value
    if traj.terminated || isnothing(traj.bootstrap_value)
        # No bootstrapping for terminated episodes
        delta = traj.rewards[end] - traj.values[end]
    else
        # Bootstrap for truncated or rollout-limited trajectories
        delta = traj.rewards[end] + gamma * traj.bootstrap_value - traj.values[end]
    end

    advantages[end] = delta

    # Compute advantages for earlier steps using the standard GAE recursion
    #TODO: @turbo?
    for i in (n - 1):-1:1
        delta = traj.rewards[i] + gamma * traj.values[i + 1] - traj.values[i]
        advantages[i] = delta + gamma * gae_lambda * advantages[i + 1]
    end

    return nothing
end
