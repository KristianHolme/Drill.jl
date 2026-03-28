# All buffer type definitions

"""
    RolloutBuffer

On-policy rollout storage: stacked observations, actions, rewards, GAE advantages, returns, old log-probs and values for one PPO update.

Typically constructed via `RolloutBuffer(observation_space, action_space, gae_lambda, gamma, n_steps, n_envs)`.
"""
struct RolloutBuffer{T <: AbstractFloat, O, A} <: AbstractBuffer
    observations::Array{O}
    actions::Array{A}
    rewards::Vector{T}
    advantages::Vector{T}
    returns::Vector{T}
    logprobs::Vector{T}
    values::Vector{T}
    gae_lambda::T
    gamma::T
    n_steps::Int
    n_envs::Int
end

mutable struct Trajectory{T <: AbstractFloat, O, A}
    observations::Vector{O}
    actions::Vector{A}
    rewards::Vector{T}
    logprobs::Vector{T}
    values::Vector{T}
    terminated::Bool
    truncated::Bool
    bootstrap_value::Union{Nothing, T}  # Value of the next state for truncated episodes
end

"""
    OffPolicyTrajectory{T,O,A}

A mutable container for storing a single trajectory of off-policy experience data including observations, actions, rewards, and termination information.
"""
mutable struct OffPolicyTrajectory{T <: AbstractFloat, O, A}
    observations::Vector{O}
    actions::Vector{A}
    rewards::Vector{T}
    terminated::Bool
    truncated::Bool
    truncated_observation::Union{Nothing, O}
end

"""
    ReplayBuffer{T,O,OBS,AC}

A circular buffer for storing multiple trajectories of off-policy experience data, used for replay-based learning algorithms.

# Truncation Logic
- If `terminated = true`, then there should be no `truncated_observation`
- If `truncated = true`, then there should be a `truncated_observation`  
- If `terminated = false` and `truncated = false`, then we stopped in the middle of an episode, so there should be a `truncated_observation`
"""
struct ReplayBuffer{T, O, OBS, AC}
    observation_space::O
    action_space::Box
    observations::CircularBuffer{OBS}
    actions::CircularBuffer{AC}
    rewards::CircularBuffer{T}
    terminated::CircularBuffer{Bool}
    truncated::CircularBuffer{Bool}
    truncated_observations::CircularBuffer{Union{Nothing, OBS}}
end
