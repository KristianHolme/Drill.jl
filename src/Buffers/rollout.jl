# RolloutBuffer-specific implementations

Base.length(rb::RolloutBuffer) = rb.n_steps * rb.n_envs

function RolloutBuffer(
        observation_space::AbstractSpace,
        action_space::AbstractSpace,
        n_steps::Int,
        n_envs::Int;
        dtype::Type{T} = Float32,
    ) where {T <: AbstractFloat}
    total_steps = n_steps * n_envs
    observations = Array{eltype(observation_space)}(undef, size(observation_space)..., total_steps)
    actions = Array{eltype(action_space)}(undef, size(action_space)..., total_steps)
    buffer = RolloutBuffer{T, typeof(observation_space), typeof(action_space), typeof(observations), typeof(actions)}(
        observation_space,
        action_space,
        observations,
        actions,
        zeros(T, total_steps),
        zeros(T, total_steps),
        zeros(T, total_steps),
        zeros(T, total_steps),
        zeros(T, total_steps),
        zeros(Bool, total_steps),
        zeros(Bool, total_steps),
        zeros(T, total_steps),
        n_steps,
        n_envs,
    )
    reset!(buffer)
    return buffer
end

function reset!(rollout_buffer::RolloutBuffer)
    fill!(rollout_buffer.observations, zero(eltype(rollout_buffer.observations)))
    fill!(rollout_buffer.actions, zero(eltype(rollout_buffer.actions)))
    fill!(rollout_buffer.rewards, 0)
    fill!(rollout_buffer.advantages, 0)
    fill!(rollout_buffer.returns, 0)
    fill!(rollout_buffer.logprobs, 0)
    fill!(rollout_buffer.values, 0)
    fill!(rollout_buffer.terminateds, false)
    fill!(rollout_buffer.truncateds, false)
    fill!(rollout_buffer.bootstrap_values, 0)
    return nothing
end

observation_space(buffer::RolloutBuffer) = buffer.observation_space
action_space(buffer::RolloutBuffer) = buffer.action_space

"""
    step_indices(buffer::RolloutBuffer, t) -> UnitRange

Indices of the `n_envs` transitions stored for step `t`.
"""
step_indices(buffer::RolloutBuffer, t::Int) = ((t - 1) * buffer.n_envs + 1):(t * buffer.n_envs)

"""
    store_step!(buffer, t, observations, actions, rewards, logprobs, values, terminateds, truncateds)

Write the transitions of all envs at step `t`. `observations` and `actions` are batched,
with the env index last.
"""
function store_step!(
        buffer::RolloutBuffer, t::Int, observations::AbstractArray, actions::AbstractArray,
        rewards::AbstractVector, logprobs::AbstractVector, values::AbstractVector,
        terminateds::AbstractVector{Bool}, truncateds::AbstractVector{Bool},
    )
    inds = step_indices(buffer, t)
    selectdim(buffer.observations, ndims(buffer.observations), inds) .= observations
    selectdim(buffer.actions, ndims(buffer.actions), inds) .= actions
    buffer.rewards[inds] .= rewards
    buffer.logprobs[inds] .= logprobs
    buffer.values[inds] .= values
    buffer.terminateds[inds] .= terminateds
    buffer.truncateds[inds] .= truncateds
    return buffer
end

"""
    compute_gae!(buffer::RolloutBuffer, gamma, gae_lambda)

Compute generalized advantage estimates and returns for all envs at once, going backwards
over steps. A terminated transition has no next value; a truncated transition, or the last
step of an env that did not finish, uses its bootstrap value.
"""
function compute_gae!(buffer::RolloutBuffer, gamma::T, gae_lambda::T) where {T <: AbstractFloat}
    n_envs, n_steps = buffer.n_envs, buffer.n_steps
    rewards, values, advantages = buffer.rewards, buffer.values, buffer.advantages
    for t in n_steps:-1:1, e in 1:n_envs
        i = (t - 1) * n_envs + e
        if buffer.terminateds[i]
            next_value = zero(T)
            next_advantage = zero(T)
        elseif buffer.truncateds[i] || t == n_steps
            next_value = buffer.bootstrap_values[i]
            next_advantage = zero(T)
        else
            next_value = values[i + n_envs]
            next_advantage = advantages[i + n_envs]
        end
        delta = rewards[i] + gamma * next_value - values[i]
        advantages[i] = delta + gamma * gae_lambda * next_advantage
    end
    buffer.returns .= advantages .+ values
    return buffer
end
