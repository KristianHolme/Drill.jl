# Utility functions for actor-critic layers

noise(layer::ContinuousActorCriticLayer{<:Any, <:Any, N, <:Any, <:Any, <:Any, <:Any, <:Any}) where {N <: AbstractNoise} = N()
noise(layer::DiscreteActorCriticLayer{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any}) = NoNoise()

observation_space(layer::AbstractActorCriticLayer) = layer.observation_space
action_space(layer::AbstractActorCriticLayer) = layer.action_space
