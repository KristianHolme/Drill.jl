# Agent methods

#takes vector of observations
#TODO: adjust name?
"""
    get_action_and_values(agent::Agent, observations::AbstractVector) -> (actions, values, logprobs)

Get actions, values, and log probabilities for a vector of observations.

# Arguments
- `agent::Agent`: The agent
- `observations::AbstractVector`: Vector of observations

# Returns  
- `actions`: Vector of actions (processed for environment use)
- `values`: Vector of value estimates
- `logprobs`: Vector of log probabilities
"""
function get_action_and_values(
        agent::Agent{<:AbstractActorCriticLayer, ALG, <:AbstractActionAdapter, <:AbstractRNG, <:AbstractTrainingLogger, <:Any},
        observations::AbstractVector
    ) where {ALG <: AbstractAlgorithm}
    #TODO add !to name?
    layer = agent.layer
    train_state = agent.train_state
    ps = train_state.parameters
    st = train_state.states
    batched_obs = batch(observations, observation_space(layer))
    dev = get_device(ps)
    if dev !== nothing
        batched_obs = batched_obs |> dev
    end
    actions_batched, values, logprobs, st = layer(batched_obs, ps, st)
    @reset train_state.states = st
    agent.train_state = train_state
    actions = _actions_to_vector(actions_batched)
    return actions, values, logprobs
end

"""
    predict_values(agent::Agent, observations::AbstractVector) -> Vector

Predict value estimates for a vector of observations.

# Arguments
- `agent::Agent`: The agent
- `observations::AbstractVector`: Vector of observations

# Returns
- `Vector`: Value estimates for each observation
"""
function predict_values(
        agent::Agent{<:AbstractActorCriticLayer, ALG, <:AbstractActionAdapter, <:AbstractRNG, <:AbstractTrainingLogger, <:Any},
        observations::AbstractVector
    ) where {ALG <: AbstractAlgorithm}
    #TODO add !to name?
    layer = agent.layer
    train_state = agent.train_state
    ps = train_state.parameters
    st = train_state.states
    # Convert observations vector to batched matrix for policy
    batched_obs = batch(observations, observation_space(layer))
    dev = get_device(ps)
    if dev !== nothing
        batched_obs = batched_obs |> dev
    end
    values, st = predict_values(layer, batched_obs, ps, st)
    @reset train_state.states = st
    agent.train_state = train_state
    return values
end

"""
    predict_actions(agent::Agent, observations::AbstractVector; kwargs...) -> Vector

Predict actions for a vector of observations, processed for environment use.

# Arguments
- `agent::Agent`: The agent
- `observations::AbstractVector`: Vector of observations
- `deterministic::Bool=false`: Whether to use deterministic actions
- `rng::AbstractRNG=agent.rng`: Random number generator
- `raw::Bool=false`: Whether to return raw actions (not processed for environment). Not supported for generic Agent.

# Returns
- `Vector`: Actions processed for environment use (e.g., 0-based for Discrete spaces), or raw actions if `raw=true` (if supported)
"""
function predict_actions(
        agent::Agent{<:AbstractActorCriticLayer, ALG, <:AbstractActionAdapter, <:AbstractRNG, <:AbstractTrainingLogger, <:Any},
        observations::AbstractVector;
        deterministic::Bool = false,
        rng::AbstractRNG = agent.rng,
        raw::Bool = false
    ) where {ALG <: AbstractAlgorithm}
    if raw
        error("Agent does not support raw actions. Use an algorithm/agent that supports raw actions.")
    end
    #TODO add !to name?
    layer = agent.layer
    train_state = agent.train_state
    ps = train_state.parameters
    st = train_state.states
    batched_obs = batch(observations, observation_space(layer))
    dev = get_device(ps)
    if dev !== nothing
        batched_obs = batched_obs |> dev
    end
    actions_batched, st = predict_actions(layer, batched_obs, ps, st; deterministic = deterministic, rng = rng)
    @reset train_state.states = st
    agent.train_state = train_state
    actions_vec = _actions_to_vector(actions_batched)
    adapter = agent.action_adapter
    actions = to_env.(Ref(adapter), actions_vec, Ref(action_space(layer)))
    return actions
end

function _actions_to_vector(actions::AbstractVector)
    return collect(actions)
end

function _actions_to_vector(actions::AbstractArray)
    return collect(eachslice(actions, dims = ndims(actions)))
end

# Abstract methods for all agents
function save_policy_params_and_state(agent::AbstractAgent, path::AbstractString; suffix::String = ".jld2")
    error("save_policy_params_and_state not implemented for $(typeof(agent))")
end

function load_policy_params_and_state(agent::AbstractAgent, path::AbstractString; suffix::String = ".jld2")
    error("load_policy_params_and_state not implemented for $(typeof(agent))")
end

function make_optimizer(optimizer_type::Type{<:Optimisers.AbstractRule}, alg::AbstractAlgorithm)
    return optimizer_type(alg.learning_rate)
end


# Implementation for unified Agent (always save from CPU)
function save_policy_params_and_state(
        agent::Agent{<:AbstractActorCriticLayer, ALG, <:AbstractActionAdapter, <:AbstractRNG, <:AbstractTrainingLogger, <:Any},
        path::AbstractString;
        suffix::String = ".jld2"
    ) where {ALG <: AbstractAlgorithm}
    file_path = endswith(path, suffix) ? path : path * suffix
    @info "Saving policy, parameters, and state to $file_path"
    agent_cpu = agent |> cpu_device()
    save(
        file_path, Dict(
            "layer" => agent_cpu.layer,
            "parameters" => agent_cpu.train_state.parameters,
            "states" => agent_cpu.train_state.states,
            "aux" => agent_cpu.aux
        )
    )
    return file_path
end
