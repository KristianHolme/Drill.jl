# Lux integration for actor-critic layers

#TODO: add ent_coef as parameter for Q-value critics?
function Lux.initialparameters(rng::AbstractRNG, model::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any})
    feats_params = (feature_extractor = Lux.initialparameters(rng, model.feature_extractor),)
    head_params = (
        actor_head = Lux.initialparameters(rng, model.actor_head),
        critic_head = Lux.initialparameters(rng, model.critic_head),
        log_std = model.log_std_init *
            ones(typeof(model.log_std_init), size(model.action_space)..., 1),
    )
    params = merge(feats_params, head_params)
    return params
end

function Lux.initialparameters(rng::AbstractRNG, model::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any})
    feats_params = (
        actor_feature_extractor = Lux.initialparameters(rng, model.feature_extractor),
        critic_feature_extractor = Lux.initialparameters(rng, model.feature_extractor),
    )
    head_params = (
        actor_head = Lux.initialparameters(rng, model.actor_head),
        critic_head = Lux.initialparameters(rng, model.critic_head),
        log_std = model.log_std_init *
            ones(typeof(model.log_std_init), size(model.action_space)..., 1),
    )
    params = merge(feats_params, head_params)
    return params
end

function Lux.initialparameters(rng::AbstractRNG, model::DiscreteActorCriticModel{<:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any})
    feats_params = (feature_extractor = Lux.initialparameters(rng, model.feature_extractor),)
    head_params = (
        actor_head = Lux.initialparameters(rng, model.actor_head),
        critic_head = Lux.initialparameters(rng, model.critic_head),
    )
    params = merge(feats_params, head_params)
    return params
end

function Lux.initialparameters(rng::AbstractRNG, model::DiscreteActorCriticModel{<:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any})
    feats_params = (
        actor_feature_extractor = Lux.initialparameters(rng, model.feature_extractor),
        critic_feature_extractor = Lux.initialparameters(rng, model.feature_extractor),
    )
    head_params = (
        actor_head = Lux.initialparameters(rng, model.actor_head),
        critic_head = Lux.initialparameters(rng, model.critic_head),
    )
    params = merge(feats_params, head_params)
    return params
end

function Lux.initialstates(rng::AbstractRNG, model::Union{ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any}, DiscreteActorCriticModel{<:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any}})
    feats_states = (feature_extractor = Lux.initialstates(rng, model.feature_extractor),)
    head_states = (
        actor_head = Lux.initialstates(rng, model.actor_head),
        critic_head = Lux.initialstates(rng, model.critic_head),
    )
    states = merge(feats_states, head_states)
    return states
end

function Lux.initialstates(rng::AbstractRNG, model::Union{ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any}, DiscreteActorCriticModel{<:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any}})
    feats_states = (
        actor_feature_extractor = Lux.initialstates(rng, model.feature_extractor),
        critic_feature_extractor = Lux.initialstates(rng, model.feature_extractor),
    )
    head_states = (
        actor_head = Lux.initialstates(rng, model.actor_head),
        critic_head = Lux.initialstates(rng, model.critic_head),
    )
    states = merge(feats_states, head_states)
    return states
end

#TODO: add ent_coef as parameter for Q-value critics?
function Lux.parameterlength(model::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any})
    feats_len = Lux.parameterlength(model.feature_extractor)
    head_len = Lux.parameterlength(model.actor_head) +
        Lux.parameterlength(model.critic_head)
    total_len = feats_len + head_len + prod(model.action_space.shape)
    return total_len
end

function Lux.parameterlength(model::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any})
    feats_len = Lux.parameterlength(model.feature_extractor) +
        Lux.parameterlength(model.feature_extractor)
    head_len = Lux.parameterlength(model.actor_head) +
        Lux.parameterlength(model.critic_head)
    total_len = feats_len + head_len + prod(model.action_space.shape)
    return total_len
end

function Lux.parameterlength(model::DiscreteActorCriticModel{<:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any})
    feats_len = Lux.parameterlength(model.feature_extractor)
    head_len = Lux.parameterlength(model.actor_head) +
        Lux.parameterlength(model.critic_head)
    return feats_len + head_len
end

function Lux.parameterlength(model::DiscreteActorCriticModel{<:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any})
    feats_len = Lux.parameterlength(model.feature_extractor) +
        Lux.parameterlength(model.feature_extractor)
    head_len = Lux.parameterlength(model.actor_head) +
        Lux.parameterlength(model.critic_head)
    return feats_len + head_len
end

function Lux.statelength(model::Union{ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any}, DiscreteActorCriticModel{<:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any}})
    feats_len = Lux.statelength(model.feature_extractor)
    head_len = Lux.statelength(model.actor_head) +
        Lux.statelength(model.critic_head)
    return feats_len + head_len
end

function Lux.statelength(model::Union{ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any}, DiscreteActorCriticModel{<:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any}})
    feats_len = Lux.statelength(model.feature_extractor) +
        Lux.statelength(model.feature_extractor)
    head_len = Lux.statelength(model.actor_head) +
        Lux.statelength(model.critic_head)
    return feats_len + head_len
end
