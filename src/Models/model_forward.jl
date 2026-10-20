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

"""
    actor_features(model::ContinuousActorCriticModel, obs, ps, st) -> (feats, st)
    critic_features(model::ContinuousActorCriticModel, obs, ps, st) -> (feats, st)

Features for the actor or the critic head alone. `ps` and `st` need only the parameters
and states of that feature extractor: `feature_extractor` with shared features,
`actor_feature_extractor` or `critic_feature_extractor` with separate features.
"""
function actor_features(model::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SharedFeatures}, obs::AbstractArray, ps, st)
    feats, feats_st = model.feature_extractor(obs, ps.feature_extractor, st.feature_extractor)
    return feats, merge(st, (; feature_extractor = feats_st))
end

function actor_features(model::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SeparateFeatures}, obs::AbstractArray, ps, st)
    feats, feats_st = model.feature_extractor(obs, ps.actor_feature_extractor, st.actor_feature_extractor)
    return feats, merge(st, (; actor_feature_extractor = feats_st))
end

critic_features(model::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SharedFeatures}, obs::AbstractArray, ps, st) =
    actor_features(model, obs, ps, st)

function critic_features(model::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SeparateFeatures}, obs::AbstractArray, ps, st)
    feats, feats_st = model.feature_extractor(obs, ps.critic_feature_extractor, st.critic_feature_extractor)
    return feats, merge(st, (; critic_feature_extractor = feats_st))
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
    values, critic_st = apply_q_head(model.critic_head, feats, actions, ps.critic_head, st.critic_head)
    st = merge(st, (; critic_head = critic_st))
    return values, st
end

# Q heads take features and actions as two arguments rather than one `(feats, actions)`
# tuple. When one of them is constant and the other is differentiated (the critic as seen
# by the actor loss), a tuple of both has mixed activity, which Enzyme rejects unless
# runtime activity is on. The heads built by `get_critic_head` (a `BranchLayer` of `Chain`s
# that start with a `Parallel`) are applied part by part, so no such tuple is formed.
# Other layers get the tuple.
apply_q_head(layer, feats, actions, ps, st) = layer((feats, actions), ps, st)

function apply_q_head(layer::Lux.BranchLayer, feats, actions, ps, st)
    ys, sts = _apply_q_branches(values(layer.layers), feats, actions, values(ps), values(st))
    return layer.fusion(ys...), NamedTuple{keys(layer.layers)}(sts)
end

_apply_q_branches(::Tuple{}, feats, actions, ps, st) = ((), ())
function _apply_q_branches(layers::Tuple, feats, actions, ps, st)
    y, s = apply_q_head(first(layers), feats, actions, first(ps), first(st))
    ys, sts = _apply_q_branches(Base.tail(layers), feats, actions, Base.tail(ps), Base.tail(st))
    return (y, ys...), (s, sts...)
end

function apply_q_head(layer::Lux.Chain, feats, actions, ps, st)
    layers = values(layer.layers)
    y, first_st = apply_q_head(first(layers), feats, actions, first(values(ps)), first(values(st)))
    y, rest_st = _apply_chain(Base.tail(layers), y, Base.tail(values(ps)), Base.tail(values(st)))
    return y, NamedTuple{keys(layer.layers)}((first_st, rest_st...))
end

_apply_chain(::Tuple{}, x, ps, st) = (x, ())
function _apply_chain(layers::Tuple, x, ps, st)
    y, s = first(layers)(x, first(ps), first(st))
    y, rest_st = _apply_chain(Base.tail(layers), y, Base.tail(ps), Base.tail(st))
    return y, (s, rest_st...)
end

function apply_q_head(layer::Lux.Parallel{<:Function, <:NamedTuple{(:layer_1, :layer_2)}}, feats, actions, ps, st)
    y_feats, feats_st = layer.layers.layer_1(feats, ps.layer_1, st.layer_1)
    y_actions, actions_st = layer.layers.layer_2(actions, ps.layer_2, st.layer_2)
    return layer.connection(y_feats, y_actions), (layer_1 = feats_st, layer_2 = actions_st)
end

# Helpers for batched distribution API
_distribution_type(::ContinuousActorCriticModel{<:Any, <:Any, <:Any, VCritic}) = BatchedDiagGaussian()
_distribution_type(::ContinuousActorCriticModel{<:Any, <:Any, <:Any, QCritic}) = BatchedSquashedDiagGaussian()
