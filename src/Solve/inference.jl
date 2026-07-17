function _action_slices(actions_batched)
    actions_batched = cpu_device()(actions_batched)
    if actions_batched isa AbstractVector
        return collect(actions_batched)
    end
    return collect(eachslice(actions_batched, dims = ndims(actions_batched)))
end

function _host_vector(x)
    return collect(cpu_device()(x))
end

_stored_action(::AbstractActionAdapter, action, ::Box) = action
_stored_action(adapter::AbstractActionAdapter, action, space::Discrete) = to_env(adapter, action, space)

function _env_action(cache::RLCache, action)
    return to_env(cache.adapter, action, action_space(cache.prob.env))
end

function get_action_and_values(cache::RLCache, observations::AbstractVector)
    model = cache.model
    ps = parameters(cache)
    st = rollout_inference_state(states(cache))
    obs_batch = batch(observations, observation_space(cache.prob.env))
    dev = current_device(ps)
    obs_batch = canonicalize_device_batch(dev, obs_batch |> dev)
    actions_batched, values, logprobs, st = execute_rollout_action_values(
        dev,
        cache,
        obs_batch,
        ps,
        st,
        cache.rng,
    )
    set_states!(cache, st)
    action_space_ = action_space(cache.prob.env)
    raw_actions = _action_slices(actions_batched)
    actions = _stored_action.(Ref(cache.adapter), raw_actions, Ref(action_space_))
    return actions, _host_vector(values), _host_vector(logprobs)
end

function predict_actions(
        cache::RLCache,
        observations::AbstractVector;
        deterministic::Bool = false,
        rng::AbstractRNG = cache.rng,
        raw::Bool = false,
    )
    model = cache.model
    ps = parameters(cache)
    st = rollout_inference_state(states(cache))
    obs_batch = batch(observations, observation_space(cache.prob.env))
    dev = current_device(ps)
    obs_batch = canonicalize_device_batch(dev, obs_batch |> dev)
    actions_batched, st = execute_rollout_predict_actions(
        dev,
        cache,
        obs_batch,
        ps,
        st;
        deterministic,
        rng,
    )
    set_states!(cache, st)
    actions = _action_slices(actions_batched)
    if raw
        return actions
    end
    return _stored_action.(Ref(cache.adapter), actions, Ref(action_space(cache.prob.env)))
end

function predict_actions_raw(cache::RLCache, observations::AbstractVector)
    return predict_actions(cache, observations; raw = true)
end

function predict_values(cache::RLCache, observations::AbstractVector)
    model = cache.model
    ps = parameters(cache)
    st = rollout_inference_state(states(cache))
    obs_batch = batch(observations, observation_space(cache.prob.env))
    dev = current_device(ps)
    obs_batch = canonicalize_device_batch(dev, obs_batch |> dev)
    values, st = execute_rollout_predict_values(dev, cache, obs_batch, ps, st)
    set_states!(cache, st)
    return _host_vector(values)
end
