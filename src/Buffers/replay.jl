# ReplayBuffer-specific implementations

function OffPolicyTrajectory(observation_space::AbstractSpace, action_space::AbstractSpace)
    obs_type = typeof(rand(observation_space))
    action_type = typeof(rand(action_space))
    obs_scalar_type = eltype(observation_space)
    action_scalar_type = eltype(action_space)
    @assert obs_scalar_type == action_scalar_type "Observation and action types must be the same"
    T = obs_scalar_type
    return OffPolicyTrajectory{T, obs_type, action_type}(obs_type[], action_type[], T[], false, false, nothing)
end

Base.length(traj::OffPolicyTrajectory) = length(traj.rewards)

function ReplayBuffer(observation_space::O, action_space::Box, capacity::Int) where {O}
    OBS = typeof(rand(observation_space))
    AC = typeof(rand(action_space))
    obs_scalar_type = eltype(observation_space)
    action_scalar_type = eltype(action_space)
    @assert obs_scalar_type == action_scalar_type "Observation and action types must be the same"
    T = obs_scalar_type
    return ReplayBuffer{T, O, OBS, AC}(
        observation_space,
        action_space,
        CircularBuffer{OBS}(capacity),
        CircularBuffer{AC}(capacity),
        CircularBuffer{T}(capacity),
        CircularBuffer{Bool}(capacity),
        CircularBuffer{Bool}(capacity),
        CircularBuffer{Union{Nothing, OBS}}(capacity)
    )
end

observation_space(buffer::ReplayBuffer) = buffer.observation_space
action_space(buffer::ReplayBuffer) = buffer.action_space

function Base.length(buffer::ReplayBuffer)
    obs_len = length(buffer.observations)
    action_len = length(buffer.actions)
    reward_len = length(buffer.rewards)
    terminated_len = length(buffer.terminated)
    truncated_len = length(buffer.truncated)
    truncated_obs_len = length(buffer.truncated_observations)
    @assert allequal([obs_len, action_len, reward_len, terminated_len, truncated_len, truncated_obs_len]) "All buffers must have the same length"
    return obs_len
end
Base.size(buffer::ReplayBuffer) = length(buffer)

function DataStructures.isfull(buffer::ReplayBuffer)
    obs_full = isfull(buffer.observations)
    action_full = isfull(buffer.actions)
    reward_full = isfull(buffer.rewards)
    terminated_full = isfull(buffer.terminated)
    truncated_full = isfull(buffer.truncated)
    truncated_obs_full = isfull(buffer.truncated_observations)
    @assert allequal([obs_full, action_full, reward_full, terminated_full, truncated_full, truncated_obs_full]) "All buffers must have the same length"
    return obs_full
end

function Base.empty!(buffer::ReplayBuffer)
    empty!(buffer.observations)
    empty!(buffer.actions)
    empty!(buffer.rewards)
    empty!(buffer.terminated)
    empty!(buffer.truncated)
    empty!(buffer.truncated_observations)
    return nothing
end

function Base.isempty(buffer::ReplayBuffer)
    obs_empty = isempty(buffer.observations)
    action_empty = isempty(buffer.actions)
    reward_empty = isempty(buffer.rewards)
    terminated_empty = isempty(buffer.terminated)
    truncated_empty = isempty(buffer.truncated)
    truncated_obs_empty = isempty(buffer.truncated_observations)
    @assert allequal(
        [
            true, obs_empty, action_empty, reward_empty, terminated_empty,
            truncated_empty, truncated_obs_empty,
        ]
    ) "All buffers must have the same length"
    return obs_empty
end

function DataStructures.capacity(buffer::ReplayBuffer)
    obs_cap = capacity(buffer.observations)
    action_cap = capacity(buffer.actions)
    reward_cap = capacity(buffer.rewards)
    terminated_cap = capacity(buffer.terminated)
    truncated_cap = capacity(buffer.truncated)
    truncated_obs_cap = capacity(buffer.truncated_observations)
    @assert allequal([obs_cap, action_cap, reward_cap, terminated_cap, truncated_cap, truncated_obs_cap]) "All buffers must have the same capacity"
    return obs_cap
end

#TODO: make tests
function Base.push!(buffer::ReplayBuffer, traj::OffPolicyTrajectory)
    push!(buffer.observations, traj.observations...)
    push!(buffer.actions, traj.actions...)
    push!(buffer.rewards, traj.rewards...)
    vec_terminated = fill(false, length(traj.observations))
    vec_terminated[end] = traj.terminated
    push!(buffer.terminated, vec_terminated...)
    vec_truncated = fill(false, length(traj.observations))
    vec_truncated[end] = traj.truncated
    push!(buffer.truncated, vec_truncated...)
    OBS = typeof(traj.observations[1])
    vec_truncated_obs = Vector{Union{Nothing, OBS}}(nothing, length(traj.observations))
    vec_truncated_obs[end] = traj.truncated_observation
    push!(buffer.truncated_observations, vec_truncated_obs...)
    return nothing
end

function get_data_loader(buffer::ReplayBuffer{T, O, OBS, AC}, batch_size::Int, batches::Int, shuffle::Bool, parallel::Bool, rng::AbstractRNG) where {T, O, OBS, AC}
    buffer_size = length(buffer)
    samples = batch_size * batches
    sample_inds = sample(rng, 1:buffer_size, samples, replace = true)

    obs_sample = batch(buffer.observations[sample_inds], observation_space(buffer))
    action_sample = batch(buffer.actions[sample_inds], action_space(buffer))
    reward_sample = buffer.rewards[sample_inds]
    terminated_sample = buffer.terminated[sample_inds]
    truncated_sample = buffer.truncated[sample_inds]
    truncated_obs_sample = buffer.truncated_observations[sample_inds]

    next_obs_sample = Vector{OBS}(undef, samples)
    for i in 1:samples
        if !terminated_sample[i]
            next_obs = if isnothing(truncated_obs_sample[i])
                #step is in the middle of a rollout, so we just take the next observation
                buffer.observations[sample_inds[i] + 1]
            else
                #step is at end of rollout, or truncated by episode time limit
                truncated_obs_sample[i]
            end
            next_obs_sample[i] = next_obs
        else
            #dummy value to have next_obs be same length as the other arrays
            #take first value to get shape, and multiply with NaN to get all NaNs
            #TODO is there a cleaner simpler way to do this?
            next_obs_sample[i] = buffer.observations[1] * NaN
        end
    end
    next_obs_sample = batch(next_obs_sample, observation_space(buffer))
    #check that all elements are assigned
    @assert all(x -> isassigned(next_obs_sample, x), eachindex(next_obs_sample))

    return DataLoader(
        (
            observations = obs_sample, actions = action_sample,
            rewards = reward_sample, terminated = terminated_sample,
            truncated = truncated_sample, next_observations = next_obs_sample,
        );
        batchsize = batch_size, shuffle, parallel, rng
    )
end
