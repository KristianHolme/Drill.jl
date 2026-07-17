# RolloutBuffer-specific implementations

Base.length(rb::RolloutBuffer) = rb.n_steps * rb.n_envs

#TODO: fix types here
function RolloutBuffer(
        observation_space::AbstractSpace,
        action_space::AbstractSpace,
        n_steps::Int,
        n_envs::Int;
        dtype::Type{T} = Float32,
    ) where {T <: AbstractFloat}
    total_steps = n_steps * n_envs
    obs_eltype = eltype(observation_space)
    action_eltype = eltype(action_space)
    observations = Array{obs_eltype}(undef, size(observation_space)..., total_steps)
    actions = Array{action_eltype}(undef, size(action_space)..., total_steps)
    rewards = Vector{T}(undef, total_steps)
    advantages = Vector{T}(undef, total_steps)
    returns = Vector{T}(undef, total_steps)
    logprobs = Vector{T}(undef, total_steps)
    values = Vector{T}(undef, total_steps)
    terminateds = Vector{Bool}(undef, total_steps)
    truncateds = Vector{Bool}(undef, total_steps)
    bootstrap_values = Vector{Union{Nothing, T}}(undef, total_steps)
    episode_ends = Int[]
    return RolloutBuffer{T, typeof(observation_space), typeof(action_space), obs_eltype, action_eltype}(
        observation_space,
        action_space,
        observations,
        actions,
        rewards,
        advantages,
        returns,
        logprobs,
        values,
        terminateds,
        truncateds,
        bootstrap_values,
        episode_ends,
        n_steps,
        n_envs,
    )
end

function RolloutBuffer(
        observation_space::AbstractSpace,
        action_space::AbstractSpace,
        ::AbstractFloat,
        ::AbstractFloat,
        n_steps::Int,
        n_envs::Int,
    )
    return RolloutBuffer(observation_space, action_space, n_steps, n_envs)
end

function reset!(rollout_buffer::RolloutBuffer)
    rollout_buffer.observations .= 0
    rollout_buffer.actions .= 0
    rollout_buffer.rewards .= 0
    rollout_buffer.advantages .= 0
    rollout_buffer.returns .= 0
    rollout_buffer.logprobs .= 0
    rollout_buffer.values .= 0
    rollout_buffer.terminateds .= false
    rollout_buffer.truncateds .= false
    rollout_buffer.bootstrap_values .= nothing
    empty!(rollout_buffer.episode_ends)
    return nothing
end

function pack_trajectories!(
        rollout_buffer::RolloutBuffer,
        trajectories::AbstractVector{<:Trajectory},
    )
    reset!(rollout_buffer)
    obs_space = observation_space(rollout_buffer)
    act_space = action_space(rollout_buffer)
    traj_lengths = length.(trajectories)
    positions = cumsum([1; traj_lengths])
    for (i, traj) in enumerate(trajectories)
        #transfer data to the Rolloutbuffer
        traj_inds = positions[i]:(positions[i + 1] - 1)
        # @debug "traj_inds: $(traj_inds)"
        selectdim(rollout_buffer.observations, ndims(obs_space) + 1, traj_inds) .= batch(traj.observations, obs_space)
        selectdim(rollout_buffer.actions, ndims(act_space) + 1, traj_inds) .= batch(traj.actions, act_space)
        rollout_buffer.rewards[traj_inds] .= traj.rewards
        rollout_buffer.logprobs[traj_inds] .= traj.logprobs
        rollout_buffer.values[traj_inds] .= traj.values
        rollout_buffer.terminateds[traj_inds] .= false
        rollout_buffer.truncateds[traj_inds] .= false
        rollout_buffer.bootstrap_values[traj_inds] .= nothing
        rollout_buffer.terminateds[last(traj_inds)] = traj.terminated
        rollout_buffer.truncateds[last(traj_inds)] = traj.truncated
        rollout_buffer.bootstrap_values[last(traj_inds)] = traj.bootstrap_value
        push!(rollout_buffer.episode_ends, last(traj_inds))
    end
    return rollout_buffer
end

observation_space(buffer::RolloutBuffer) = buffer.observation_space
action_space(buffer::RolloutBuffer) = buffer.action_space

function compute_gae!(rollout_buffer::RolloutBuffer, gamma::T, gae_lambda::T) where {T <: AbstractFloat}
    start = 1
    for ending in rollout_buffer.episode_ends
        traj_inds = start:ending
        compute_advantages!(
            @view(rollout_buffer.advantages[traj_inds]),
            @view(rollout_buffer.rewards[traj_inds]),
            @view(rollout_buffer.values[traj_inds]),
            rollout_buffer.terminateds[ending],
            rollout_buffer.bootstrap_values[ending],
            gamma,
            gae_lambda,
        )
        rollout_buffer.returns[traj_inds] .= rollout_buffer.advantages[traj_inds] .+
            rollout_buffer.values[traj_inds]
        start = ending + 1
    end
    return rollout_buffer
end
