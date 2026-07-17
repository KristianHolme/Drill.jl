# Actor-critic layer constructor functions

function ContinuousActorCriticLayer(
        observation_space::Union{Discrete, Box{T}},
        action_space::Box{T};
        log_std_init = T(0),
        hidden_dims = [64, 64],
        activation = tanh,
        shared_features::Bool = true,
        critic_type::CriticType = VCritic()
    ) where {T}

    feature_extractor = get_feature_extractor(observation_space)
    latent_dim = size(observation_space) |> prod
    #TODO: make this bias init work for different types
    bias_init = zeros32

    hidden_init = OrthogonalInitializer{T}(sqrt(T(2)))
    actor_init = OrthogonalInitializer{T}(T(0.01))
    value_init = OrthogonalInitializer{T}(T(1.0))
    actor_head = get_actor_head(
        latent_dim, action_space, hidden_dims, activation,
        bias_init, hidden_init, actor_init
    )
    critic_head = get_critic_head(
        latent_dim, action_space, hidden_dims, activation,
        bias_init, hidden_init, value_init, critic_type
    )

    # Choose feature sharing type based on boolean flag
    F = shared_features ? SharedFeatures : SeparateFeatures

    return ContinuousActorCriticLayer{
        typeof(observation_space),
        typeof(action_space),
        StateIndependantNoise,
        typeof(critic_type),
        F,
        typeof(feature_extractor),
        typeof(actor_head),
        typeof(critic_head),
        typeof(log_std_init),
    }(
        observation_space,
        action_space,
        feature_extractor,
        actor_head,
        critic_head,
        log_std_init
    )
end

function DiscreteActorCriticLayer(
        observation_space::Union{Discrete, Box},
        action_space::Discrete; hidden_dims = [64, 64], activation = tanh,
        shared_features::Bool = true
    )
    feature_extractor = get_feature_extractor(observation_space)
    latent_dim = size(observation_space) |> prod
    #TODO: make this bias init work for different types
    bias_init = zeros32

    hidden_init = OrthogonalInitializer{Float32}(sqrt(Float32(2)))
    actor_init = OrthogonalInitializer{Float32}(Float32(0.01))
    value_init = OrthogonalInitializer{Float32}(Float32(1.0))
    actor_head = get_actor_head(
        latent_dim, action_space, hidden_dims, activation,
        bias_init, hidden_init, actor_init
    )
    critic_head = get_critic_head(
        latent_dim, action_space, hidden_dims, activation,
        bias_init, hidden_init, value_init, VCritic()
    )

    # Choose feature sharing type based on boolean flag
    F = shared_features ? SharedFeatures : SeparateFeatures

    return DiscreteActorCriticLayer{
        typeof(observation_space),
        typeof(action_space),
        F,
        typeof(feature_extractor),
        typeof(actor_head),
        typeof(critic_head),
    }(
        observation_space, action_space, feature_extractor,
        actor_head, critic_head
    )
end


# Unified constructor name
"""
    ActorCriticLayer(observation_space, action_space::Box; kwargs...) -> ContinuousActorCriticLayer
    ActorCriticLayer(observation_space, action_space::Discrete; kwargs...) -> DiscreteActorCriticLayer

Unified constructor that forwards to `ContinuousActorCriticLayer` or `DiscreteActorCriticLayer` depending on the action space type.
"""
ActorCriticLayer(observation_space::Union{Discrete, Box}, action_space::Box; kwargs...) =
    ContinuousActorCriticLayer(observation_space, action_space; kwargs...)
ActorCriticLayer(observation_space::Union{Discrete, Box}, action_space::Discrete; kwargs...) =
    DiscreteActorCriticLayer(observation_space, action_space; kwargs...)
