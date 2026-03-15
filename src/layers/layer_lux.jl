# Lux integration for actor-critic layers

#TODO: add ent_coef as parameter for Q-value critics?
function Lux.initialparameters(rng::AbstractRNG, layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any})
    feats_params = (feature_extractor = Lux.initialparameters(rng, layer.feature_extractor),)
    head_params = (
        actor_head = Lux.initialparameters(rng, layer.actor_head),
        critic_head = Lux.initialparameters(rng, layer.critic_head),
        log_std = layer.log_std_init *
            ones(typeof(layer.log_std_init), size(layer.action_space)..., 1),
    )
    params = merge(feats_params, head_params)
    return params
end

function Lux.initialparameters(rng::AbstractRNG, layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any})
    feats_params = (
        actor_feature_extractor = Lux.initialparameters(rng, layer.feature_extractor),
        critic_feature_extractor = Lux.initialparameters(rng, layer.feature_extractor),
    )
    head_params = (
        actor_head = Lux.initialparameters(rng, layer.actor_head),
        critic_head = Lux.initialparameters(rng, layer.critic_head),
        log_std = layer.log_std_init *
            ones(typeof(layer.log_std_init), size(layer.action_space)..., 1),
    )
    params = merge(feats_params, head_params)
    return params
end

function Lux.initialparameters(rng::AbstractRNG, layer::DiscreteActorCriticLayer{<:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any})
    feats_params = (feature_extractor = Lux.initialparameters(rng, layer.feature_extractor),)
    head_params = (
        actor_head = Lux.initialparameters(rng, layer.actor_head),
        critic_head = Lux.initialparameters(rng, layer.critic_head),
    )
    params = merge(feats_params, head_params)
    return params
end

function Lux.initialparameters(rng::AbstractRNG, layer::DiscreteActorCriticLayer{<:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any})
    feats_params = (
        actor_feature_extractor = Lux.initialparameters(rng, layer.feature_extractor),
        critic_feature_extractor = Lux.initialparameters(rng, layer.feature_extractor),
    )
    head_params = (
        actor_head = Lux.initialparameters(rng, layer.actor_head),
        critic_head = Lux.initialparameters(rng, layer.critic_head),
    )
    params = merge(feats_params, head_params)
    return params
end

function Lux.initialstates(rng::AbstractRNG, layer::Union{ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any}, DiscreteActorCriticLayer{<:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any}})
    feats_states = (feature_extractor = Lux.initialstates(rng, layer.feature_extractor),)
    head_states = (
        actor_head = Lux.initialstates(rng, layer.actor_head),
        critic_head = Lux.initialstates(rng, layer.critic_head),
    )
    states = merge(feats_states, head_states)
    return states
end

function Lux.initialstates(rng::AbstractRNG, layer::Union{ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any}, DiscreteActorCriticLayer{<:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any}})
    feats_states = (
        actor_feature_extractor = Lux.initialstates(rng, layer.feature_extractor),
        critic_feature_extractor = Lux.initialstates(rng, layer.feature_extractor),
    )
    head_states = (
        actor_head = Lux.initialstates(rng, layer.actor_head),
        critic_head = Lux.initialstates(rng, layer.critic_head),
    )
    states = merge(feats_states, head_states)
    return states
end

#TODO: add ent_coef as parameter for Q-value critics?
function Lux.parameterlength(layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any})
    feats_len = Lux.parameterlength(layer.feature_extractor)
    head_len = Lux.parameterlength(layer.actor_head) +
        Lux.parameterlength(layer.critic_head)
    total_len = feats_len + head_len + prod(layer.action_space.shape)
    return total_len
end

function Lux.parameterlength(layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any})
    feats_len = Lux.parameterlength(layer.feature_extractor) +
        Lux.parameterlength(layer.feature_extractor)
    head_len = Lux.parameterlength(layer.actor_head) +
        Lux.parameterlength(layer.critic_head)
    total_len = feats_len + head_len + prod(layer.action_space.shape)
    return total_len
end

function Lux.parameterlength(layer::DiscreteActorCriticLayer{<:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any})
    feats_len = Lux.parameterlength(layer.feature_extractor)
    head_len = Lux.parameterlength(layer.actor_head) +
        Lux.parameterlength(layer.critic_head)
    return feats_len + head_len
end

function Lux.parameterlength(layer::DiscreteActorCriticLayer{<:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any})
    feats_len = Lux.parameterlength(layer.feature_extractor) +
        Lux.parameterlength(layer.feature_extractor)
    head_len = Lux.parameterlength(layer.actor_head) +
        Lux.parameterlength(layer.critic_head)
    return feats_len + head_len
end

function Lux.statelength(layer::Union{ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any}, DiscreteActorCriticLayer{<:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any}})
    feats_len = Lux.statelength(layer.feature_extractor)
    head_len = Lux.statelength(layer.actor_head) +
        Lux.statelength(layer.critic_head)
    return feats_len + head_len
end

function Lux.statelength(layer::Union{ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any}, DiscreteActorCriticLayer{<:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any}})
    feats_len = Lux.statelength(layer.feature_extractor) +
        Lux.statelength(layer.feature_extractor)
    head_len = Lux.statelength(layer.actor_head) +
        Lux.statelength(layer.critic_head)
    return feats_len + head_len
end
