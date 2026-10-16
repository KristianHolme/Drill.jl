# ReplayBuffer-specific implementations

function ReplayBuffer(observation_space::AbstractSpace, action_space::Box, capacity::Int)
    @assert capacity > 0 "capacity must be positive"
    T = eltype(action_space)
    @assert eltype(observation_space) == T "Observation and action types must be the same"
    observations = Array{eltype(observation_space)}(undef, size(observation_space)..., capacity)
    next_observations = similar(observations)
    actions = Array{T}(undef, size(action_space)..., capacity)
    return ReplayBuffer{T, typeof(observation_space), typeof(action_space), typeof(observations), typeof(actions)}(
        observation_space,
        action_space,
        observations,
        next_observations,
        actions,
        zeros(T, capacity),
        zeros(Bool, capacity),
        zeros(Bool, capacity),
        1,
        0,
    )
end

observation_space(buffer::ReplayBuffer) = buffer.observation_space
action_space(buffer::ReplayBuffer) = buffer.action_space

Base.length(buffer::ReplayBuffer) = buffer.count
Base.size(buffer::ReplayBuffer) = length(buffer)
capacity(buffer::ReplayBuffer) = length(buffer.rewards)
isfull(buffer::ReplayBuffer) = buffer.count == capacity(buffer)
Base.isempty(buffer::ReplayBuffer) = buffer.count == 0

function Base.empty!(buffer::ReplayBuffer)
    buffer.position = 1
    buffer.count = 0
    return buffer
end

"""
    add_transitions!(buffer::ReplayBuffer, observations, actions, rewards, terminated, truncated, next_observations)

Append one transition per env. `observations`, `actions` and `next_observations` are
batched with the env index last; the others are vectors.
"""
function add_transitions!(
        buffer::ReplayBuffer, observations::AbstractArray, actions::AbstractArray,
        rewards::AbstractVector, terminated::AbstractVector{Bool},
        truncated::AbstractVector{Bool}, next_observations::AbstractArray,
    )
    cap = capacity(buffer)
    obs_dim = ndims(buffer.observations)
    act_dim = ndims(buffer.actions)
    for j in eachindex(rewards)
        i = buffer.position
        selectdim(buffer.observations, obs_dim, i) .= selectdim(observations, obs_dim, j)
        selectdim(buffer.next_observations, obs_dim, i) .= selectdim(next_observations, obs_dim, j)
        selectdim(buffer.actions, act_dim, i) .= selectdim(actions, act_dim, j)
        buffer.rewards[i] = rewards[j]
        buffer.terminated[i] = terminated[j]
        buffer.truncated[i] = truncated[j]
        buffer.position = mod1(i + 1, cap)
        buffer.count = min(buffer.count + 1, cap)
    end
    return buffer
end

function _gather(x::AbstractArray, inds)
    return collect(selectdim(x, ndims(x), inds))
end

"""
    sample_batch(buffer::ReplayBuffer, n, rng) -> NamedTuple

`n` transitions drawn uniformly with replacement.
"""
function sample_batch(buffer::ReplayBuffer, n::Int, rng::AbstractRNG)
    @assert !isempty(buffer) "Cannot sample from an empty buffer"
    inds = rand(rng, 1:length(buffer), n)
    return (
        observations = _gather(buffer.observations, inds),
        actions = _gather(buffer.actions, inds),
        rewards = buffer.rewards[inds],
        terminated = buffer.terminated[inds],
        truncated = buffer.truncated[inds],
        next_observations = _gather(buffer.next_observations, inds),
    )
end

function get_data_loader(buffer::ReplayBuffer, batch_size::Int, batches::Int, shuffle::Bool, parallel::Bool, rng::AbstractRNG)
    data = sample_batch(buffer, batch_size * batches, rng)
    return DataLoader(data; batchsize = batch_size, shuffle, parallel, rng)
end
