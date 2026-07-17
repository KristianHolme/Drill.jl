# Concrete actor-critic layer type definitions

"""
    ContinuousActorCriticLayer

Actor–critic Lux model for continuous (`Box`) or discrete observations with continuous actions: feature extractor, stochastic actor (with learnable log-std where applicable), and value or Q heads according to `critic_type`.

Use `ContinuousActorCriticLayer(observation_space, action_space::Box; kwargs...)` to build a layer.
"""
struct ContinuousActorCriticLayer{
        O <: AbstractSpace,
        A <: Box,
        N <: AbstractNoise,
        C <: CriticType,
        F <: FeatureSharing,
        FE <: AbstractLuxLayer,
        AH <: AbstractLuxLayer,
        CH <: AbstractLuxLayer,
        LS,
    } <: AbstractActorCriticLayer
    observation_space::O
    action_space::A
    feature_extractor::FE
    actor_head::AH
    critic_head::CH
    log_std_init::LS
end

"""
    DiscreteActorCriticLayer

Actor–critic Lux model for discrete actions: feature extractor, categorical policy head, and value head.

Use `DiscreteActorCriticLayer(observation_space, action_space::Discrete; kwargs...)`.
"""
struct DiscreteActorCriticLayer{O <: AbstractSpace, A <: Discrete, F <: FeatureSharing, FE <: AbstractLuxLayer, AH <: AbstractLuxLayer, CH <: AbstractLuxLayer} <: AbstractActorCriticLayer
    observation_space::O
    action_space::A
    feature_extractor::FE
    actor_head::AH
    critic_head::CH
end
