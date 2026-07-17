# Trajectory-specific functions

function Trajectory{T}(observation_space::AbstractSpace, action_space::AbstractSpace) where {T <: AbstractFloat}
    obs_type = typeof(rand(observation_space))
    action_type = typeof(rand(action_space))
    observations = obs_type[]
    actions = action_type[]
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
