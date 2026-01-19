# Lux integration for actor-critic layers

#TODO: add ent_coef as parameter for Q-value critics?
function Lux.initialparameters(rng::AbstractRNG, policy::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any})
    feats_params = (feature_extractor = Lux.initialparameters(rng, policy.feature_extractor),)
    head_params = (
        actor_head = Lux.initialparameters(rng, policy.actor_head),
        critic_head = Lux.initialparameters(rng, policy.critic_head),
        log_std = policy.log_std_init *
            ones(typeof(policy.log_std_init), size(policy.action_space)),
    )
    params = merge(feats_params, head_params)
    return params
end

function Lux.initialparameters(rng::AbstractRNG, policy::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any})
    feats_params = (
        actor_feature_extractor = Lux.initialparameters(rng, policy.feature_extractor),
        critic_feature_extractor = Lux.initialparameters(rng, policy.feature_extractor),
    )
    head_params = (
        actor_head = Lux.initialparameters(rng, policy.actor_head),
        critic_head = Lux.initialparameters(rng, policy.critic_head),
        log_std = policy.log_std_init *
            ones(typeof(policy.log_std_init), size(policy.action_space)),
    )
    params = merge(feats_params, head_params)
    return params
end

function Lux.initialparameters(rng::AbstractRNG, policy::DiscreteActorCriticLayer{<:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any})
    feats_params = (feature_extractor = Lux.initialparameters(rng, policy.feature_extractor),)
    head_params = (
        actor_head = Lux.initialparameters(rng, policy.actor_head),
        critic_head = Lux.initialparameters(rng, policy.critic_head),
    )
    params = merge(feats_params, head_params)
    return params
end

function Lux.initialparameters(rng::AbstractRNG, policy::DiscreteActorCriticLayer{<:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any})
    feats_params = (
        actor_feature_extractor = Lux.initialparameters(rng, policy.feature_extractor),
        critic_feature_extractor = Lux.initialparameters(rng, policy.feature_extractor),
    )
    head_params = (
        actor_head = Lux.initialparameters(rng, policy.actor_head),
        critic_head = Lux.initialparameters(rng, policy.critic_head),
    )
    params = merge(feats_params, head_params)
    return params
end

function Lux.initialstates(rng::AbstractRNG, policy::Union{ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any}, DiscreteActorCriticLayer{<:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any}})
    feats_states = (feature_extractor = Lux.initialstates(rng, policy.feature_extractor),)
    head_states = (
        actor_head = Lux.initialstates(rng, policy.actor_head),
        critic_head = Lux.initialstates(rng, policy.critic_head),
    )
    states = merge(feats_states, head_states)
    return states
end

function Lux.initialstates(rng::AbstractRNG, policy::Union{ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any}, DiscreteActorCriticLayer{<:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any}})
    feats_states = (
        actor_feature_extractor = Lux.initialstates(rng, policy.feature_extractor),
        critic_feature_extractor = Lux.initialstates(rng, policy.feature_extractor),
    )
    head_states = (
        actor_head = Lux.initialstates(rng, policy.actor_head),
        critic_head = Lux.initialstates(rng, policy.critic_head),
    )
    states = merge(feats_states, head_states)
    return states
end

#TODO: add ent_coef as parameter for Q-value critics?
function Lux.parameterlength(policy::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any})
    feats_len = Lux.parameterlength(policy.feature_extractor)
    head_len = Lux.parameterlength(policy.actor_head) +
        Lux.parameterlength(policy.critic_head)
    total_len = feats_len + head_len + prod(policy.action_space.shape)
    return total_len
end

function Lux.parameterlength(policy::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any})
    feats_len = Lux.parameterlength(policy.feature_extractor) +
        Lux.parameterlength(policy.feature_extractor)
    head_len = Lux.parameterlength(policy.actor_head) +
        Lux.parameterlength(policy.critic_head)
    total_len = feats_len + head_len + prod(policy.action_space.shape)
    return total_len
end

function Lux.parameterlength(policy::DiscreteActorCriticLayer{<:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any})
    feats_len = Lux.parameterlength(policy.feature_extractor)
    head_len = Lux.parameterlength(policy.actor_head) +
        Lux.parameterlength(policy.critic_head)
    return feats_len + head_len
end

function Lux.parameterlength(policy::DiscreteActorCriticLayer{<:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any})
    feats_len = Lux.parameterlength(policy.feature_extractor) +
        Lux.parameterlength(policy.feature_extractor)
    head_len = Lux.parameterlength(policy.actor_head) +
        Lux.parameterlength(policy.critic_head)
    return feats_len + head_len
end

function Lux.statelength(policy::Union{ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any}, DiscreteActorCriticLayer{<:Any, <:Any, SharedFeatures, <:Any, <:Any, <:Any}})
    feats_len = Lux.statelength(policy.feature_extractor)
    head_len = Lux.statelength(policy.actor_head) +
        Lux.statelength(policy.critic_head)
    return feats_len + head_len
end

function Lux.statelength(policy::Union{ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any}, DiscreteActorCriticLayer{<:Any, <:Any, SeparateFeatures, <:Any, <:Any, <:Any}})
    feats_len = Lux.statelength(policy.feature_extractor) +
        Lux.statelength(policy.feature_extractor)
    head_len = Lux.statelength(policy.actor_head) +
        Lux.statelength(policy.critic_head)
    return feats_len + head_len
end
