module Models

import Lux
using Lux: AbstractLuxLayer, Chain, Dense, ReshapeLayer, orthogonal, zeros32
using Random: AbstractRNG, default_rng

import DrillInterface: AbstractSpace, Box, Discrete, action_space, observation_space
using DrillInterface: batch

using ..DrillDistributions: BatchedCategorical, BatchedDiagGaussian, BatchedSquashedDiagGaussian,
    logpdf, entropy, mode

include("types.jl")
include("model_types.jl")
include("model_utils.jl")
include("model_helpers.jl")
include("model_constructors.jl")
include("model_lux.jl")
include("model_forward.jl")
include("model_methods.jl")

export AbstractModel, AbstractActorCriticModel, AbstractNoise, AbstractWeightInitializer
export CriticType, QCritic, VCritic
export FeatureSharing, SharedFeatures, SeparateFeatures
export StateIndependantNoise, StateDependentNoise, NoNoise
export ContinuousActorCriticModel, DiscreteActorCriticModel, ActorCriticModel
export OrthogonalInitializer
export predict_actions, predict_values, evaluate_actions, action_log_prob
export extract_features, get_actions_from_features, get_values_from_features
export observation_space, action_space

end
