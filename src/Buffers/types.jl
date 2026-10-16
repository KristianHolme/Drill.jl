abstract type AbstractBuffer end
abstract type OnPolicyBuffer <: AbstractBuffer end
abstract type OffPolicyBuffer <: AbstractBuffer end

"""
    RolloutBuffer

On-policy rollout storage for one PPO update: observations, actions, rewards, log-probs,
values, done flags, bootstrap values, and the GAE advantages and returns computed from them.

Construct with `RolloutBuffer(observation_space, action_space, n_steps, n_envs)`. Every
array has one entry per transition, `n_envs * n_steps` in all, with the transition of env
`e` at step `t` at index `(t - 1) * n_envs + e`. The collector writes the `n_envs`
transitions of each step as one contiguous block.

`bootstrap_values[i]` is the value of the state after transition `i` and is used only
where an episode is cut short: when `truncateds[i]` is set, or at the last step of the
rollout for an env that did not finish.
"""
struct RolloutBuffer{T <: AbstractFloat, S, AS, OA <: AbstractArray, AA <: AbstractArray} <: OnPolicyBuffer
    observation_space::S
    action_space::AS
    observations::OA
    actions::AA
    rewards::Vector{T}
    advantages::Vector{T}
    returns::Vector{T}
    logprobs::Vector{T}
    values::Vector{T}
    terminateds::Vector{Bool}
    truncateds::Vector{Bool}
    bootstrap_values::Vector{T}
    n_steps::Int
    n_envs::Int
end

"""
    ReplayBuffer

Off-policy transition storage as a ring of preallocated arrays: observations, actions,
rewards, done flags and next observations, `capacity` transitions in all. Once full, new
transitions overwrite the oldest.

`next_observations` holds the real next observation of every transition, including the
last observation of a finished episode, so sampling never needs a placeholder.
"""
mutable struct ReplayBuffer{T <: AbstractFloat, O, A, OA <: AbstractArray, AA <: AbstractArray} <: OffPolicyBuffer
    const observation_space::O
    const action_space::A
    const observations::OA
    const next_observations::OA
    const actions::AA
    const rewards::Vector{T}
    const terminated::Vector{Bool}
    const truncated::Vector{Bool}
    # Index the next transition is written to
    position::Int
    # Number of stored transitions
    count::Int
end
