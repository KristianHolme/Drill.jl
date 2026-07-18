# High-level actor-critic layer methods

function predict_actions(model::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any}, obs::AbstractArray, ps, st; deterministic::Bool = false, rng::AbstractRNG = default_rng())
    actor_feats, critic_feats, st = extract_features(model, obs, ps, st)
    action_means, st = get_actions_from_features(model, actor_feats, ps, st)
    d = _distribution_type(model)
    if deterministic
        actions = mode(d, action_means)
    else
        actions = rand(rng, d, action_means, ps.log_std)
    end
    return actions, st
end

function predict_actions(model::DiscreteActorCriticModel, obs::AbstractArray, ps, st; deterministic::Bool = false, rng::AbstractRNG = default_rng())
    actor_feats, critic_feats, st = extract_features(model, obs, ps, st)
    action_logits, st = get_actions_from_features(model, actor_feats, ps, st)
    probs = Lux.softmax(action_logits)
    d = BatchedCategorical()
    if deterministic
        actions_onehot = mode(d, probs)
    else
        actions_onehot = rand(rng, d, probs)
    end
    return actions_onehot, st
end

function evaluate_actions(
        model::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any},
        obs::AbstractArray,
        actions::AbstractArray,
        ps,
        st,
    )
    actor_feats, critic_feats, st = extract_features(model, obs, ps, st)
    new_action_means, st = get_actions_from_features(model, actor_feats, ps, st)
    values, st = get_values_from_features(model, critic_feats, ps, st)
    d = _distribution_type(model)
    log_probs = logpdf(d, actions, new_action_means, ps.log_std)
    entropies = entropy(d, new_action_means, ps.log_std)
    return evaluate_actions_returns(model, values, vec(log_probs), vec(entropies), st)
end

function evaluate_actions_returns(::ContinuousActorCriticModel{<:Any, <:Any, <:Any, QCritic}, values, log_probs, entropies, st)
    return values, log_probs, entropies, st #dont return vec(values) as values is a matrix
end
function evaluate_actions_returns(::ContinuousActorCriticModel, values, log_probs, entropies, st)
    return vec(values), log_probs, entropies, st
end

function evaluate_actions(model::DiscreteActorCriticModel, obs::AbstractArray, actions::AbstractMatrix, ps, st)
    actor_feats, critic_feats, st = extract_features(model, obs, ps, st)
    new_action_logits, st = get_actions_from_features(model, actor_feats, ps, st)
    values, st = get_values_from_features(model, critic_feats, ps, st)
    probs = Lux.softmax(new_action_logits)
    d = BatchedCategorical()
    log_probs = logpdf(d, actions, probs)
    entropies = entropy(d, probs)
    return vec(values), vec(log_probs), vec(entropies), st
end

function predict_values(model::AbstractActorCriticModel, obs::AbstractArray, ps, st)
    actor_feats, critic_feats, st = extract_features(model, obs, ps, st)
    values, st = get_values_from_features(model, critic_feats, ps, st)
    return vec(values), st
end

function predict_values(model::ContinuousActorCriticModel{<:Any, <:Any, N, QCritic, <:Any, <:Any, <:Any, <:Any}, obs::AbstractArray, actions::AbstractArray, ps, st) where {N <: AbstractNoise}
    actor_feats, critic_feats, st = extract_features(model, obs, ps, st)
    values, st = get_values_from_features(model, critic_feats, actions, ps, st)
    return values, st #dont return vec(values) as this is a matrix
end

function action_log_prob(model::ContinuousActorCriticModel, obs::AbstractArray, ps, st; rng::AbstractRNG = default_rng())
    actor_feats, _, st = extract_features(model, obs, ps, st)
    action_means, st = get_actions_from_features(model, actor_feats, ps, st)
    d = _distribution_type(model)
    actions = rand(rng, d, action_means, ps.log_std)
    log_probs = logpdf(d, actions, action_means, ps.log_std)
    return actions, vec(log_probs), st
end
