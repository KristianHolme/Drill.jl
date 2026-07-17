# Utility functions for actor-critic layers

noise(model::ContinuousActorCriticModel{<:Any, <:Any, N, <:Any, <:Any, <:Any, <:Any, <:Any}) where {N <: AbstractNoise} = N()
noise(model::DiscreteActorCriticModel{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any}) = NoNoise()

observation_space(model::AbstractActorCriticModel) = model.observation_space
action_space(model::AbstractActorCriticModel) = model.action_space
