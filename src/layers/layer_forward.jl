# Forward pass implementations for actor-critic layers

function (layer::ContinuousActorCriticLayer)(obs::AbstractArray, ps, st)
    actor_feats, critic_feats, st = extract_features(layer, obs, ps, st)
    action_means, st = get_actions_from_features(layer, actor_feats, ps, st)
    values, st = get_values_from_features(layer, critic_feats, ps, st)
    d = _distribution_type(layer)
    rng = Random.default_rng()
    actions = rand(rng, d, action_means, ps.log_std)
    log_probs = logpdf(d, actions, action_means, ps.log_std)
    return actions, vec(values), vec(log_probs), st
end

function (layer::ContinuousActorCriticLayer{<:Any, <:Any, N, QCritic, <:Any, <:Any, <:Any, <:Any})(
        obs::AbstractArray,
        actions::AbstractArray, ps, st
    ) where {N <: AbstractNoise}
    actor_feats, critic_feats, st = extract_features(layer, obs, ps, st)
    action_means, st = get_actions_from_features(layer, actor_feats, ps, st)
    values, st = get_values_from_features(layer, critic_feats, actions, ps, st)
    d = _distribution_type(layer)
    rng = Random.default_rng()
    actions = rand(rng, d, action_means, ps.log_std)
    log_probs = logpdf(d, actions, action_means, ps.log_std)
    return actions, values, vec(log_probs), st
end

function (layer::DiscreteActorCriticLayer)(obs::AbstractArray, ps, st)
    actor_feats, critic_feats, st = extract_features(layer, obs, ps, st)
    action_logits, st = get_actions_from_features(layer, actor_feats, ps, st)
    values, st = get_values_from_features(layer, critic_feats, ps, st)
    probs = Lux.softmax(action_logits)
    d = BatchedCategorical()
    rng = Random.default_rng()
    actions_onehot = rand(rng, d, probs)
    actions = onehotbatch_to_discrete(actions_onehot, action_space(layer))
    log_probs = logpdf(d, actions_onehot, probs)
    return actions, vec(values), vec(log_probs), st
end

# Type-stable feature extraction using dispatch
function extract_features(layer::ContinuousActorCriticLayer{O, A, N, C, SharedFeatures, FE, AH, CH, LS}, obs::AbstractArray, ps, st) where {O, A, N, C, FE, AH, CH, LS}
    feats, feats_st = layer.feature_extractor(obs, ps.feature_extractor, st.feature_extractor)
    actor_feats = feats
    critic_feats = feats
    st = merge(st, (; feature_extractor = feats_st))
    return actor_feats, critic_feats, st
end

function extract_features(layer::ContinuousActorCriticLayer{O, A, N, C, SeparateFeatures, FE, AH, CH, LS}, obs::AbstractArray, ps, st) where {O, A, N, C, FE, AH, CH, LS}
    actor_feats, actor_feats_st = layer.feature_extractor(obs, ps.actor_feature_extractor, st.actor_feature_extractor)
    critic_feats, critic_feats_st = layer.feature_extractor(obs, ps.critic_feature_extractor, st.critic_feature_extractor)
    st = merge(st, (; actor_feature_extractor = actor_feats_st, critic_feature_extractor = critic_feats_st))
    return actor_feats, critic_feats, st
end

# For DiscreteActorCriticLayer (3 type parameters)
function extract_features(layer::DiscreteActorCriticLayer{O, A, SharedFeatures, FE, AH, CH}, obs::AbstractArray, ps, st) where {O, A, FE, AH, CH}
    feats, feats_st = layer.feature_extractor(obs, ps.feature_extractor, st.feature_extractor)
    actor_feats = feats
    critic_feats = feats
    st = merge(st, (; feature_extractor = feats_st))
    return actor_feats, critic_feats, st
end

function extract_features(layer::DiscreteActorCriticLayer{O, A, SeparateFeatures, FE, AH, CH}, obs::AbstractArray, ps, st) where {O, A, FE, AH, CH}
    actor_feats, actor_feats_st = layer.feature_extractor(obs, ps.actor_feature_extractor, st.actor_feature_extractor)
    critic_feats, critic_feats_st = layer.feature_extractor(obs, ps.critic_feature_extractor, st.critic_feature_extractor)
    st = merge(st, (; actor_feature_extractor = actor_feats_st, critic_feature_extractor = critic_feats_st))
    return actor_feats, critic_feats, st
end

# Direct calls to concrete layer fields keep inference intact

function get_actions_from_features(layer::AbstractActorCriticLayer, feats::AbstractArray, ps, st)
    # Use function barrier to isolate type instability
    actions, actor_st = layer.actor_head(copy(feats), ps.actor_head, st.actor_head)
    st = merge(st, (; actor_head = actor_st))
    return actions, st
end

function get_values_from_features(layer::AbstractActorCriticLayer, feats::AbstractArray, ps, st)
    # Use function barrier to isolate type instability
    values, critic_st = layer.critic_head(feats, ps.critic_head, st.critic_head)
    st = merge(st, (; critic_head = critic_st))
    return values, st
end

function get_values_from_features(layer::ContinuousActorCriticLayer{<:Any, <:Any, N, QCritic, <:Any, <:Any, <:Any, <:Any}, feats::AbstractArray, actions::AbstractArray, ps, st) where {N <: AbstractNoise}
    if ndims(actions) == 1
        actions = batch(actions, action_space(layer))
    end
    inputs = vcat(feats, actions)
    # Use function barrier to isolate type instability
    #TODO: runtime dispatch
    values, critic_st = layer.critic_head(inputs, ps.critic_head, st.critic_head)
    st = merge(st, (; critic_head = critic_st))
    return values, st
end

# Helpers for batched distribution API
_distribution_type(::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, VCritic}) = BatchedDiagGaussian()
_distribution_type(::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, QCritic}) = BatchedSquashedDiagGaussian()
