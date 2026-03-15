function current_device(x)
    dev = get_device(x)
    if isnothing(dev)
        return cpu_device()
    end
    return dev
end

canonicalize_device_batch(dev, obs_batch) = obs_batch

reactant_cache_entry_count(::Any) = 0

rollout_inference_state(st) = st
deployment_inference_state(st) = st

function rollout_action_values_kernel(layer, obs_batch, ps, st, rng)
    return layer(obs_batch, ps, st; rng)
end

function rollout_predict_actions_kernel(
        layer,
        obs_batch,
        ps,
        st;
        deterministic::Bool,
        rng,
    )
    return predict_actions(layer, obs_batch, ps, st; deterministic, rng)
end

function rollout_predict_actions_deterministic_kernel(layer, obs_batch, ps, st)
    return predict_actions(
        layer,
        obs_batch,
        ps,
        st;
        deterministic = true,
        rng = Random.default_rng(),
    )
end

function rollout_predict_actions_stochastic_kernel(layer, obs_batch, ps, st, rng)
    return predict_actions(layer, obs_batch, ps, st; deterministic = false, rng)
end

function rollout_predict_values_kernel(layer, obs_batch, ps, st)
    return predict_values(layer, obs_batch, ps, st)
end

function deployment_predict_actions_kernel(
        layer,
        obs_batch,
        ps,
        st;
        deterministic::Bool,
        rng,
    )
    return predict_actions(layer, obs_batch, ps, st; deterministic, rng)
end

function deployment_predict_actions_deterministic_kernel(layer, obs_batch, ps, st)
    return predict_actions(
        layer,
        obs_batch,
        ps,
        st;
        deterministic = true,
        rng = Random.default_rng(),
    )
end

function deployment_predict_actions_stochastic_kernel(layer, obs_batch, ps, st, rng)
    return predict_actions(layer, obs_batch, ps, st; deterministic = false, rng)
end

function execute_rollout_action_values(
        dev::MLDataDevices.AbstractDevice,
        agent,
        obs_batch,
        ps,
        st,
        rng,
    )
    return rollout_action_values_kernel(agent.layer, obs_batch, ps, st, rng)
end

function execute_rollout_predict_actions(
        dev::MLDataDevices.AbstractDevice,
        agent,
        obs_batch,
        ps,
        st;
        deterministic::Bool,
        rng,
    )
    if deterministic
        return rollout_predict_actions_deterministic_kernel(agent.layer, obs_batch, ps, st)
    end
    return rollout_predict_actions_stochastic_kernel(agent.layer, obs_batch, ps, st, rng)
end

function execute_rollout_predict_values(
        dev::MLDataDevices.AbstractDevice,
        agent,
        obs_batch,
        ps,
        st,
    )
    return rollout_predict_values_kernel(agent.layer, obs_batch, ps, st)
end

function execute_deployment_predict_actions(
        dev::MLDataDevices.AbstractDevice,
        layer,
        obs_batch,
        ps,
        st;
        deterministic::Bool,
        rng,
    )
    if deterministic
        return deployment_predict_actions_deterministic_kernel(layer.layer, obs_batch, ps, st)
    end
    return deployment_predict_actions_stochastic_kernel(layer.layer, obs_batch, ps, st, rng)
end
