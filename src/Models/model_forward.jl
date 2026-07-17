# Forward pass implementations for actor-critic layers

function (model::ContinuousActorCriticModel)(obs::AbstractArray, ps, st; rng::AbstractRNG = default_rng())
    actor_feats, critic_feats, st = extract_features(model, obs, ps, st)
    action_means, st = get_actions_from_features(model, actor_feats, ps, st)
    values, st = get_values_from_features(model, critic_feats, ps, st)
    d = _distribution_type(model)
    actions = rand(rng, d, action_means, ps.log_std)
    log_probs = logpdf(d, actions, action_means, ps.log_std)
    return actions, vec(values), vec(log_probs), st
end

function (model::ContinuousActorCriticModel{<:Any, <:Any, N, QCritic, <:Any, <:Any, <:Any, <:Any})(
        obs::AbstractArray,
        actions::AbstractArray, ps, st;
        rng::AbstractRNG = default_rng()
    ) where {N <: AbstractNoise}
    actor_feats, critic_feats, st = extract_features(model, obs, ps, st)
    action_means, st = get_actions_from_features(model, actor_feats, ps, st)
    values, st = get_values_from_features(model, critic_feats, actions, ps, st)
    d = _distribution_type(model)
    actions = rand(rng, d, action_means, ps.log_std)
    log_probs = logpdf(d, actions, action_means, ps.log_std)
    return actions, values, vec(log_probs), st
end

function (model::DiscreteActorCriticModel)(obs::AbstractArray, ps, st; rng::AbstractRNG = default_rng())
    actor_feats, critic_feats, st = extract_features(model, obs, ps, st)
    action_logits, st = get_actions_from_features(model, actor_feats, ps, st)
    values, st = get_values_from_features(model, critic_feats, ps, st)
    probs = Lux.softmax(action_logits)
    d = BatchedCategorical()
    actions_onehot = rand(rng, d, probs)
    log_probs = logpdf(d, actions_onehot, probs)
    return actions_onehot, vec(values), vec(log_probs), st
end

# Type-stable feature extraction using dispatch
function extract_features(model::ContinuousActorCriticModel{O, A, N, C, SharedFeatures, FE, AH, CH, LS}, obs::AbstractArray, ps, st) where {O, A, N, C, FE, AH, CH, LS}
    feats, feats_st = model.feature_extractor(obs, ps.feature_extractor, st.feature_extractor)
    actor_feats = feats
    critic_feats = feats
    st = merge(st, (; feature_extractor = feats_st))
    return actor_feats, critic_feats, st
end

function extract_features(model::ContinuousActorCriticModel{O, A, N, C, SeparateFeatures, FE, AH, CH, LS}, obs::AbstractArray, ps, st) where {O, A, N, C, FE, AH, CH, LS}
    actor_feats, actor_feats_st = model.feature_extractor(obs, ps.actor_feature_extractor, st.actor_feature_extractor)
    critic_feats, critic_feats_st = model.feature_extractor(obs, ps.critic_feature_extractor, st.critic_feature_extractor)
    st = merge(st, (; actor_feature_extractor = actor_feats_st, critic_feature_extractor = critic_feats_st))
    return actor_feats, critic_feats, st
end

# For DiscreteActorCriticModel (3 type parameters)
function extract_features(model::DiscreteActorCriticModel{O, A, SharedFeatures, FE, AH, CH}, obs::AbstractArray, ps, st) where {O, A, FE, AH, CH}
    feats, feats_st = model.feature_extractor(obs, ps.feature_extractor, st.feature_extractor)
    actor_feats = feats
    critic_feats = feats
    st = merge(st, (; feature_extractor = feats_st))
    return actor_feats, critic_feats, st
end

function extract_features(model::DiscreteActorCriticModel{O, A, SeparateFeatures, FE, AH, CH}, obs::AbstractArray, ps, st) where {O, A, FE, AH, CH}
    actor_feats, actor_feats_st = model.feature_extractor(obs, ps.actor_feature_extractor, st.actor_feature_extractor)
    critic_feats, critic_feats_st = model.feature_extractor(obs, ps.critic_feature_extractor, st.critic_feature_extractor)
    st = merge(st, (; actor_feature_extractor = actor_feats_st, critic_feature_extractor = critic_feats_st))
    return actor_feats, critic_feats, st
end

# Direct calls to concrete layer fields keep inference intact

function get_actions_from_features(model::AbstractActorCriticModel, feats::AbstractArray, ps, st)
    # Use function barrier to isolate type instability
    actions, actor_st = model.actor_head(feats, ps.actor_head, st.actor_head)
    st = merge(st, (; actor_head = actor_st))
    return actions, st
end

function get_values_from_features(model::AbstractActorCriticModel, feats::AbstractArray, ps, st)
    # Use function barrier to isolate type instability
    values, critic_st = model.critic_head(feats, ps.critic_head, st.critic_head)
    st = merge(st, (; critic_head = critic_st))
    return values, st
end

function get_values_from_features(model::ContinuousActorCriticModel{<:Any, <:Any, N, QCritic, <:Any, <:Any, <:Any, <:Any}, feats::AbstractArray, actions::AbstractArray, ps, st) where {N <: AbstractNoise}
    if ndims(actions) == 1
        actions = batch(actions, action_space(model))
    end
    values, critic_st = model.critic_head((feats, actions), ps.critic_head, st.critic_head)
    st = merge(st, (; critic_head = critic_st))
    return values, st
end

# Helpers for batched distribution API
_distribution_type(::ContinuousActorCriticModel{<:Any, <:Any, <:Any, VCritic}) = BatchedDiagGaussian()
_distribution_type(::ContinuousActorCriticModel{<:Any, <:Any, <:Any, QCritic}) = BatchedSquashedDiagGaussian()
