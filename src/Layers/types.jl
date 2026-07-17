# ------------------------------------------------------------
# Layers
# ------------------------------------------------------------
abstract type AbstractLayer <: Lux.AbstractLuxLayer end
abstract type CriticType end
@kwdef struct QCritic <: CriticType
    n_critics::Int = 2
end
struct VCritic <: CriticType end

"""
    predict_actions(layer::AbstractLayer, obs::AbstractArray, ps, st; deterministic::Bool=false) -> (actions, st)

Predict actions from batched observations.

# Arguments
- `layer::AbstractLayer`: The actor-critic layer
- `obs::AbstractArray`: Batched observations (last dimension is batch)
- `ps`: Layer parameters
- `st`: Layer state
- `deterministic::Bool=false`: Whether to use deterministic actions

# Returns
- `actions`: Vector/Array of actions (raw layer outputs, not processed for environment)
- `st`: Updated layer state

# Notes
- Input observations must be batched (matrix/array format)
- Output actions are raw layer outputs (e.g., 1-based for Discrete layers)
- Use `to_env()` to convert for environment use
"""
function predict_actions end


"""
    predict_values(layer::AbstractLayer, obs::AbstractArray, [actions::AbstractArray,] ps, st) -> (values, st)

Predict Q-values from batched observations and actions (for Q-Critic layers).

# Arguments
- `layer::AbstractLayer`: The actor-critic layer
- `obs::AbstractArray`: Batched observations (last dimension is batch)
- `actions::AbstractArray`: Batched actions (last dimension is batch) (only for Q-Critic layers)
- `ps`: Layer parameters
- `st`: Layer state

# Returns
- `values`: batched values (tuples of values for multiple Q-Critic networks)
- `st`: Updated layer state

# Notes
- Input observations and actions must be batched (matrix/array format)
- Actions should be in raw layer format (e.g., 1-based for Discrete)
"""
function predict_values end


"""
    evaluate_actions(layer::AbstractLayer, obs::AbstractArray, actions::AbstractArray, ps, st) -> (values, log_probs, entropy, st)

Evaluate given actions for batched observations.

# Arguments
- `layer::AbstractLayer`: The actor-critic layer
- `obs::AbstractArray`: Batched observations (last dimension is batch)
- `actions::AbstractArray`: Batched actions to evaluate (raw layer format)
- `ps`: Layer parameters
- `st`: Layer state

# Returns
- `values`: Vector of value estimates
- `log_probs`: Vector of log probabilities for the actions
- `entropy`: Vector of layer entropy values
- `st`: Updated layer state

# Notes
- All inputs must be batched (matrix/array format)
- Actions should be in raw layer format (e.g., 1-based for Discrete)
"""
function evaluate_actions end

"""
    action_log_prob(layer::AbstractLayer, obs::AbstractArray, ps, st) -> (actions, log_probs, st)

Sample actions and return their log probabilities from batched observations (for SAC).

# Arguments
- `layer::AbstractLayer`: The actor-critic layer
- `obs::AbstractArray`: Batched observations (last dimension is batch)
- `ps`: Layer parameters
- `st`: Layer state

# Returns
- `actions`: Vector/Array of sampled actions
- `log_probs`: Vector of log probabilities for the sampled actions
- `st`: Updated layer state

# Notes
- Input observations must be batched (matrix/array format)
- Output actions are raw layer outputs (e.g., 1-based for Discrete layers)
"""
function action_log_prob end

abstract type AbstractNoise end

struct StateIndependantNoise <: AbstractNoise end
struct StateDependentNoise <: AbstractNoise end
struct NoNoise <: AbstractNoise end

"""
    AbstractActorCriticLayer <: AbstractLayer

Abstract type for actor-critic layers. Subtypes are callable with signature:

    (layer::AbstractActorCriticLayer)(obs::AbstractArray, ps, st) -> (actions, values, log_probs, st)

Forward pass through layer: get actions, values, and log probabilities from batched observations.

# Arguments
- `obs::AbstractArray`: Batched observations (each column is one observation)
- `ps`: Layer parameters
- `st`: Layer state

# Returns
- `actions`: Vector/Array of actions (raw layer outputs)
- `values`: Vector of value estimates
- `log_probs`: Vector of log probabilities
- `st`: Updated layer state

# Notes
- Input observations must be batched (matrix/array format)
- Output actions are raw layer outputs (e.g., 1-based for Discrete layers)
"""
abstract type AbstractActorCriticLayer <: AbstractLayer end

# Abstract types for shared features parameter
abstract type FeatureSharing end
struct SharedFeatures <: FeatureSharing end
struct SeparateFeatures <: FeatureSharing end
