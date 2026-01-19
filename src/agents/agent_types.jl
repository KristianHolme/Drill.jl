# Agent type definitions

mutable struct AgentStats
    gradient_updates::Int
    steps_taken::Int
end

function add_step!(stats::AgentStats, steps::Int = 1)
    return stats.steps_taken += steps
end

function add_gradient_update!(stats::AgentStats, updates::Int = 1)
    return stats.gradient_updates += updates
end

function steps_taken(stats::AgentStats)
    return stats.steps_taken
end

function gradient_updates(stats::AgentStats)
    return stats.gradient_updates
end

function Random.seed!(agent::AbstractAgent, seed::Integer)
    Random.seed!(agent.rng, seed)
    return agent
end

"""
Auxiliary state for agents that do not require extra training-time structures.
"""
struct NoAux end

"""
Auxiliary state for Q-based actor-critic algorithms (e.g., SAC/TD3/DDPG).
Holds target critic parameters/states and entropy coefficient train state.
"""
mutable struct QAux
    Q_target_parameters::NamedTuple
    Q_target_states::NamedTuple #TODO: are these abstract types?
    ent_train_state::Lux.Training.TrainState
end

"""
Unified Agent for all algorithms.

verbose:
    0: nothing
    1: progress bar
    2: progress bar and stats
"""
mutable struct Agent{L <: AbstractActorCriticLayer, ALG <: AbstractAlgorithm, AD <: AbstractActionAdapter, R <: AbstractRNG, LG <: AbstractTrainingLogger, AUX} <: AbstractAgent
    layer::L
    algorithm::ALG
    action_adapter::AD
    train_state::Lux.Training.TrainState
    optimizer_type::Type{<:Optimisers.AbstractRule}
    stats_window::Int
    logger::LG
    verbose::Int
    rng::R
    stats::AgentStats
    aux::AUX
end

add_step!(agent::Agent, steps::Int = 1) = add_step!(agent.stats, steps)
add_gradient_update!(agent::Agent, updates::Int = 1) = add_gradient_update!(agent.stats, updates)
steps_taken(agent::Agent) = steps_taken(agent.stats)
gradient_updates(agent::Agent) = gradient_updates(agent.stats)
