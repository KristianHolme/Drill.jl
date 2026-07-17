module Layers

using Lux
using Lux: AbstractLuxLayer, Chain, ReshapeLayer, orthogonal
using Random
using LinearAlgebra

import DrillInterface: AbstractSpace, Box, Discrete, action_space, observation_space
using DrillInterface: batch

using ..DrillDistributions: BatchedCategorical, BatchedDiagGaussian, BatchedSquashedDiagGaussian,
    logpdf, entropy, mode

include("types.jl")
include("layer_types.jl")
include("layer_utils.jl")
include("layer_helpers.jl")
include("layer_constructors.jl")
include("layer_lux.jl")
include("layer_forward.jl")
include("layer_methods.jl")

export AbstractLayer, AbstractActorCriticLayer, AbstractNoise, AbstractWeightInitializer
export CriticType, QCritic, VCritic
export FeatureSharing, SharedFeatures, SeparateFeatures
export StateIndependantNoise, StateDependentNoise, NoNoise
export ContinuousActorCriticLayer, DiscreteActorCriticLayer, ActorCriticLayer
export OrthogonalInitializer
export predict_actions, predict_values, evaluate_actions, action_log_prob
export extract_features, get_actions_from_features, get_values_from_features
export observation_space, action_space

end
