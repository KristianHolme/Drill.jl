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

function compute_advantages!(
        advantages::AbstractArray,
        rewards::AbstractArray,
        values::AbstractArray,
        terminated::Bool,
        bootstrap_value::Union{Nothing, AbstractFloat},
        gamma::AbstractFloat,
        gae_lambda::AbstractFloat,
    )
    n = length(rewards)

    # For terminated episodes, no bootstrapping (bootstrap_value should be nothing)
    # For truncated episodes or rollout-limited trajectories, bootstrap with the next state value
    if terminated || isnothing(bootstrap_value)
        # No bootstrapping for terminated episodes
        delta = rewards[end] - values[end]
    else
        # Bootstrap for truncated or rollout-limited trajectories
        delta = rewards[end] + gamma * bootstrap_value - values[end]
    end

    advantages[end] = delta

    # Compute advantages for earlier steps using the standard GAE recursion
    #TODO: @turbo?
    for i in (n - 1):-1:1
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        advantages[i] = delta + gamma * gae_lambda * advantages[i + 1]
    end

    return nothing
end

function compute_advantages!(advantages::AbstractArray, traj::Trajectory, gamma::AbstractFloat, gae_lambda::AbstractFloat)
    return compute_advantages!(
        advantages,
        traj.rewards,
        traj.values,
        traj.terminated,
        traj.bootstrap_value,
        gamma,
        gae_lambda,
    )
end
