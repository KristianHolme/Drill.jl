module Drill_ReactantExt

# Inference-only Reactant compilation for rollout / deployment kernels.
# Training compilation is owned by Lux's TrainState cache via
# `Lux.Training.compute_gradients` / `single_train_step!` with `AutoEnzyme()`
# when parameters live on a `ReactantDevice`. Do not `@compile` models into
# TrainState construction.

using Drill
using Adapt
using MLDataDevices
using MLDataDevices: ReactantDevice
using Reactant: @compile

import Drill:
    deployment_predict_actions_deterministic_kernel,
    deployment_predict_actions_stochastic_kernel,
    execute_deployment_predict_actions,
    execute_rollout_action_values,
    execute_rollout_predict_actions,
    execute_rollout_predict_values,
    parameters,
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

function _get_cache(x)
    if x isa Drill.RLCache
        return x.inference_cache
    end
    return x.cache
end

function _set_cache!(x, cache)
    if x isa Drill.RLCache
        x.inference_cache = cache
    else
        x.cache = cache
    end
    return cache
end

function _runtime_model(x)
    if x isa Drill.RLCache
        return x.model
    elseif x isa Drill.NeuralPolicy
        return x.model
    end
    return x
end

function ensure_reactant_cache!(x)
    cache = _get_cache(x)
    if cache isa ReactantInferenceCache
        return cache
    end
    return _set_cache!(x, ReactantInferenceCache())
end

function cache_key(surface::Symbol, obs; mode::Symbol)
    return ReactantCompileKey(surface, typeof(obs), size(obs), mode)
end

function Drill.reactant_cache_entry_count(x::Union{Drill.RLCache, Drill.NeuralPolicy})
    cache = _get_cache(x)
    if !(cache isa ReactantInferenceCache)
        return 0
    end
    dev = if hasproperty(x, :params)
        Drill.current_device(x.params)
    else
        Drill.current_device(parameters(x))
    end
    if !(dev isa ReactantDevice)
        return 0
    end
    return length(cache.entries)
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
        cache_owner,
        obs,
        ps,
        st,
        rng,
    )
    cache = ensure_reactant_cache!(cache_owner)
    model = _runtime_model(cache_owner)
    rrng = Adapt.adapt(dev, rng)
    key = cache_key(:rollout_action_values, obs; mode = :stochastic)
    compiled = lookup_or_compile!(
        cache, key, () -> begin
            return @compile rollout_action_values_kernel(model, obs, ps, st, rrng)
        end
    )
    return compiled(model, obs, ps, st, rrng)
end

function Drill.execute_rollout_predict_actions(
        dev::ReactantDevice,
        cache_owner,
        obs,
        ps,
        st;
        deterministic::Bool,
        rng,
    )
    cache = ensure_reactant_cache!(cache_owner)
    model = _runtime_model(cache_owner)
    if deterministic
        key = cache_key(:rollout_predict_actions, obs; mode = :deterministic)
        compiled = lookup_or_compile!(
            cache, key, () -> begin
                return @compile rollout_predict_actions_deterministic_kernel(
                    model,
                    obs,
                    ps,
                    st,
                )
            end
        )
        return compiled(model, obs, ps, st)
    end

    rrng = Adapt.adapt(dev, rng)
    key = cache_key(:rollout_predict_actions, obs; mode = :stochastic)
    compiled = lookup_or_compile!(
        cache, key, () -> begin
            return @compile rollout_predict_actions_stochastic_kernel(
                model,
                obs,
                ps,
                st,
                rrng,
            )
        end
    )
    return compiled(model, obs, ps, st, rrng)
end

function Drill.execute_rollout_predict_values(
        dev::ReactantDevice,
        cache_owner,
        obs,
        ps,
        st,
    )
    cache = ensure_reactant_cache!(cache_owner)
    model = _runtime_model(cache_owner)
    key = cache_key(:rollout_predict_values, obs; mode = :deterministic)
    compiled = lookup_or_compile!(
        cache, key, () -> begin
            return @compile rollout_predict_values_kernel(model, obs, ps, st)
        end
    )
    return compiled(model, obs, ps, st)
end

function Drill.execute_deployment_predict_actions(
        dev::ReactantDevice,
        cache_owner,
        obs,
        ps,
        st;
        deterministic::Bool,
        rng,
    )
    cache = ensure_reactant_cache!(cache_owner)
    layer = _runtime_model(cache_owner)
    if deterministic
        key = cache_key(:deployment_predict_actions, obs; mode = :deterministic)
        compiled = lookup_or_compile!(
            cache, key, () -> begin
                return @compile deployment_predict_actions_deterministic_kernel(
                    layer,
                    obs,
                    ps,
                    st,
                )
            end
        )
        return compiled(layer, obs, ps, st)
    end

    rrng = Adapt.adapt(dev, rng)
    key = cache_key(:deployment_predict_actions, obs; mode = :stochastic)
    compiled = lookup_or_compile!(
        cache, key, () -> begin
            return @compile deployment_predict_actions_stochastic_kernel(
                layer,
                obs,
                ps,
                st,
                rrng,
            )
        end
    )
    return compiled(layer, obs, ps, st, rrng)
end

end
