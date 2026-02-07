# High-level actor-critic layer methods

function predict_actions(layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any}, obs::AbstractArray, ps, st; deterministic::Bool = false, rng::AbstractRNG = Random.default_rng())
    actor_feats, critic_feats, st = extract_features(layer, obs, ps, st)
    action_means, st = get_actions_from_features(layer, actor_feats, ps, st)
    log_std = ps.log_std
    #=
    if deterministic
        actions_batched = action_modes(layer, action_means, log_std)
    else
        actions_batched = action_samples(layer, action_means, log_std, rng)
    end
    actions_vec = collect(eachslice(actions_batched, dims = ndims(actions_batched)))
    return actions_vec, st
    =#
    ds = get_distributions(layer, action_means, log_std)
    if deterministic
        actions = mode.(ds)
    else
        actions = rand.(rng, ds)
    end
    return actions, st
end

function predict_actions(layer::DiscreteActorCriticLayer, obs::AbstractArray, ps, st; deterministic::Bool = false, rng::AbstractRNG = Random.default_rng())
    actor_feats, critic_feats, st = extract_features(layer, obs, ps, st)
    action_logits, st = get_actions_from_features(layer, actor_feats, ps, st)  # For discrete, these are logits
    #=
    if deterministic
        actions_batched = action_modes(layer, action_logits)
    else
        actions_batched = action_samples(layer, action_logits, rng)
    end
    actions_vec = collect(eachslice(actions_batched, dims = ndims(actions_batched)))
    return actions_vec, st
    =#
    ds = get_distributions(layer, action_logits)
    if deterministic
        actions = mode.(ds)
    else
        actions = rand.(rng, ds)
    end
    return actions, st
end

function evaluate_actions(layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any}, obs::AbstractArray{T}, actions::AbstractArray{T}, ps, st) where {T <: Real}
    actor_feats, critic_feats, st = extract_features(layer, obs, ps, st)
    new_action_means, st = get_actions_from_features(layer, actor_feats, ps, st) #runtime dispatch
    values, st = get_values_from_features(layer, critic_feats, ps, st) #runtime dispatch
    #=
    batch_std = reshape(ps.log_std, size(new_action_means)[1:end-1]..., 1)
    log_probs = logpdf(layer, actions, new_action_means, batch_std)
    entropies = entropy(layer, actions, new_action_means, batch_std)
    return evaluate_actions_returns(layer, values, log_probs, entropies, st)
    =#
    distributions = get_distributions(layer, new_action_means, ps.log_std) #runtime dispatch
    actions_vec = collect(eachslice(actions, dims = ndims(actions))) #runtime dispatch
    log_probs = logpdf.(distributions, actions_vec)
    entropies = entropy.(distributions) #runtime dispatch
    return evaluate_actions_returns(layer, values, log_probs, entropies, st)
end

function evaluate_actions_returns(::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, QCritic}, values, log_probs, entropies, st)
    return values, log_probs, entropies, st #dont return vec(values) as values is a matrix
end
function evaluate_actions_returns(::ContinuousActorCriticLayer, values, log_probs, entropies, st)
    return vec(values), log_probs, entropies, st
end

function evaluate_actions(layer::DiscreteActorCriticLayer, obs::AbstractArray, actions::AbstractArray{<:Int}, ps, st)
    actor_feats, critic_feats, st = extract_features(layer, obs, ps, st)
    new_action_logits, st = get_actions_from_features(layer, actor_feats, ps, st)  # For discrete, these are logits
    values, st = get_values_from_features(layer, critic_feats, ps, st)
    #=
    log_probs = logpdf(layer, actions, new_action_logits)
    entropies = entropy(layer, new_action_logits)
    return vec(values), log_probs, entropies, st
    =#
    ds = get_distributions(layer, new_action_logits)
    actions_vec = collect(eachslice(actions, dims = ndims(actions))) #::Vector{AbstractArray{T, ndims(actions) - 1}}
    log_probs = logpdf.(ds, actions_vec)
    entropies = entropy.(ds)
    return vec(values), log_probs, entropies, st
end

function predict_values(layer::AbstractActorCriticLayer, obs::AbstractArray, ps, st)
    actor_feats, critic_feats, st = extract_features(layer, obs, ps, st)
    values, st = get_values_from_features(layer, critic_feats, ps, st)
    return vec(values), st
end

function predict_values(layer::ContinuousActorCriticLayer{<:Any, <:Any, N, QCritic, <:Any, <:Any, <:Any, <:Any}, obs::AbstractArray, actions::AbstractArray, ps, st) where {N <: AbstractNoise}
    actor_feats, critic_feats, st = extract_features(layer, obs, ps, st)
    values, st = get_values_from_features(layer, critic_feats, actions, ps, st)
    return values, st #dont return vec(values) as this is a matrix
end

#returns vector of actions
function action_log_prob(layer::ContinuousActorCriticLayer, obs::AbstractArray, ps, st; rng::AbstractRNG = Random.default_rng())
    #TODO: fix runtime dispatch here in extract_features
    actor_feats, _, st = extract_features(layer, obs, ps, st)
    action_means, st = get_actions_from_features(layer, actor_feats, ps, st)
    log_std = ps.log_std
    #=
    actions = action_samples(layer, action_means, log_std, rng)
    log_probs = logpdf(layer, actions, action_means, log_std)
    return actions, log_probs, st #actions are batched
    =#
    ds = get_distributions(layer, action_means, log_std)
    actions = rand.(rng, ds)
    log_probs = logpdf.(ds, actions)
    # scaled_actions = scale_to_space.(actions, Ref(policy.action_space))
    return actions, log_probs, st
end
