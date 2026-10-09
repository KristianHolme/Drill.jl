module Drill_ReactantExt

# Reactant backend: rollout/deployment inference kernels and whole learner updates
# (`ppo_update`, `sac_update`) are compiled once per argument shape and cached.

using Adapt: Adapt
using Drill: Drill, deployment_predict_actions_deterministic_kernel,
    deployment_predict_actions_stochastic_kernel, parameters,
    rollout_action_values_kernel, rollout_predict_actions_deterministic_kernel,
    rollout_predict_actions_stochastic_kernel, rollout_predict_values_kernel
using Functors: fleaves
using Lux: Lux
using MLDataDevices: MLDataDevices, ReactantDevice
using Reactant: Reactant, @compile

const Enzyme = Reactant.Enzyme

struct ReactantCompileKey
    surface::Any
    input_type::Type
    input_size::Tuple
    mode::Any
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

# ------------------------------------------------------------
# Learner updates
# ------------------------------------------------------------

"""
    ReactantGradient()

Gradient backend used inside compiled updates: Enzyme reverse mode with Reactant's ABI.
"""
struct ReactantGradient end

Drill.gradient_backend(::ReactantDevice, ad) = ReactantGradient()
Drill.requires_fixed_shapes(::ReactantDevice) = true
Drill.device_rng(dev::ReactantDevice, rng) = Adapt.adapt(dev, rng)
# Optimizer hyperparameters become device numbers, so compiled updates can carry them.
function Drill.prepare_optimizer(dev::ReactantDevice, rule)
    return Lux.ReactantCompatibleOptimisers.make_reactant_compatible(rule, dev)
end

function _primal_and_aux(f::F, x, args...) where {F}
    loss, aux = f(x, args...)
    return loss, Reactant.ignore_derivatives(aux)
end

function Drill.value_and_gradient(::ReactantGradient, f::F, x, args...) where {F}
    dx = Enzyme.make_zero(x)
    _, (loss, aux) = Enzyme.autodiff(
        Enzyme.set_abi(Enzyme.ReverseWithPrimal, Reactant.ReactantABI),
        Enzyme.Const(_primal_and_aux),
        Enzyme.Duplicated,
        Enzyme.Const(f),
        Enzyme.Duplicated(x, dx),
        map(Enzyme.Const, args)...,
    )
    return loss, aux, dx
end

_array_sizes(x) = Tuple(size(l) for l in fleaves(x) if l isa AbstractArray)

function Drill.run_update(dev::ReactantDevice, cache::Drill.RLCache, update::F, args...) where {F}
    compile_cache = ensure_reactant_cache!(cache)
    key = ReactantCompileKey(update, typeof(args), _array_sizes(args), nothing)
    compiled = lookup_or_compile!(compile_cache, key, () -> @compile update(args...))
    return compiled(args...)
end

end
