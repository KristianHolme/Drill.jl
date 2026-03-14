module Drill_ReactantExt

using Drill
using Adapt
using MLDataDevices
using MLDataDevices: ReactantDevice
using Random: Random
using Reactant: @compile

import Drill:
    deployment_predict_actions_deterministic_kernel,
    deployment_predict_actions_stochastic_kernel,
    execute_deployment_predict_actions,
    execute_rollout_action_values,
    execute_rollout_predict_actions,
    execute_rollout_predict_values,
    reactant_cache_entry_count,
    rollout_action_values_kernel,
    rollout_predict_actions_deterministic_kernel,
    rollout_predict_actions_stochastic_kernel,
    rollout_predict_values_kernel

struct ReactantCompileKey
    surface::Symbol
    input_type::DataType
    input_size::Tuple
    mode::Symbol
end

mutable struct ReactantInferenceCache
    entries::Dict{ReactantCompileKey, Any}
end

MLDataDevices.isleaf(::ReactantInferenceCache) = true

function ReactantInferenceCache()
    return ReactantInferenceCache(Dict{ReactantCompileKey, Any}())
end

function ensure_reactant_cache!(x)
    if x.cache isa ReactantInferenceCache
        return x.cache
    end
    x.cache = ReactantInferenceCache()
    return x.cache
end

function cache_key(surface::Symbol, obs; mode::Symbol)
    return ReactantCompileKey(surface, typeof(obs), size(obs), mode)
end

function Drill.reactant_cache_entry_count(x::Union{Drill.Agent, Drill.NeuralPolicy})
    if !(x.cache isa ReactantInferenceCache)
        return 0
    end
    dev = if hasproperty(x, :params)
        Drill.current_device(x.params)
    else
        Drill.current_device(x.train_state.parameters)
    end
    if !(dev isa ReactantDevice)
        return 0
    end
    return length(x.cache.entries)
end

function lookup_or_compile!(cache::ReactantInferenceCache, key::ReactantCompileKey, compiler)
    if haskey(cache.entries, key)
        return cache.entries[key]
    end
    compiled = compiler()
    cache.entries[key] = compiled
    return compiled
end

function Drill.execute_rollout_action_values(
        dev::ReactantDevice,
        agent,
        obs,
        ps,
        st,
        rng,
    )
    cache = ensure_reactant_cache!(agent)
    rrng = Adapt.adapt(dev, rng)
    key = cache_key(:rollout_action_values, obs; mode = :stochastic)
    compiled = lookup_or_compile!(cache, key, () -> begin
        return @compile rollout_action_values_kernel(agent.layer, obs, ps, st, rrng)
    end)
    return compiled(agent.layer, obs, ps, st, rrng)
end

function Drill.execute_rollout_predict_actions(
        dev::ReactantDevice,
        agent,
        obs,
        ps,
        st;
        deterministic::Bool,
        rng,
    )
    cache = ensure_reactant_cache!(agent)
    if deterministic
        key = cache_key(:rollout_predict_actions, obs; mode = :deterministic)
        compiled = lookup_or_compile!(cache, key, () -> begin
            return @compile rollout_predict_actions_deterministic_kernel(
                agent.layer,
                obs,
                ps,
                st,
            )
        end)
        return compiled(agent.layer, obs, ps, st)
    end

    rrng = Adapt.adapt(dev, rng)
    key = cache_key(:rollout_predict_actions, obs; mode = :stochastic)
    compiled = lookup_or_compile!(cache, key, () -> begin
        return @compile rollout_predict_actions_stochastic_kernel(
            agent.layer,
            obs,
            ps,
            st,
            rrng,
        )
    end)
    return compiled(agent.layer, obs, ps, st, rrng)
end

function Drill.execute_rollout_predict_values(
        dev::ReactantDevice,
        agent,
        obs,
        ps,
        st,
    )
    cache = ensure_reactant_cache!(agent)
    key = cache_key(:rollout_predict_values, obs; mode = :deterministic)
    compiled = lookup_or_compile!(cache, key, () -> begin
        return @compile rollout_predict_values_kernel(agent.layer, obs, ps, st)
    end)
    return compiled(agent.layer, obs, ps, st)
end

function Drill.execute_deployment_predict_actions(
        dev::ReactantDevice,
        policy,
        obs,
        ps,
        st;
        deterministic::Bool,
        rng,
    )
    cache = ensure_reactant_cache!(policy)
    if deterministic
        key = cache_key(:deployment_predict_actions, obs; mode = :deterministic)
        compiled = lookup_or_compile!(cache, key, () -> begin
            return @compile deployment_predict_actions_deterministic_kernel(
                policy.layer,
                obs,
                ps,
                st,
            )
        end)
        return compiled(policy.layer, obs, ps, st)
    end

    rrng = Adapt.adapt(dev, rng)
    key = cache_key(:deployment_predict_actions, obs; mode = :stochastic)
    compiled = lookup_or_compile!(cache, key, () -> begin
        return @compile deployment_predict_actions_stochastic_kernel(
            policy.layer,
            obs,
            ps,
            st,
            rrng,
        )
    end)
    return compiled(policy.layer, obs, ps, st, rrng)
end

end
