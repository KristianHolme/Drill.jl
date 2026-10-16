# Deployment-time policy (actor-only wrapper)

"""
    NeuralPolicy

Lightweight inference policy holding a trained model, Lux parameters/states, the environment action space, and an [`AbstractActionAdapter`](@ref). Built via [`extract_policy`](@ref); callable on batched or single observations to produce environment actions.
"""
mutable struct NeuralPolicy{L, AD, S} <: AbstractPolicy
    model::L
    params
    states::S
    action_space
    adapter::AD
    cache
end

function NeuralPolicy(model::L, params, states::S, action_space, adapter::AD) where {L, AD, S}
    return NeuralPolicy(model, params, states, action_space, adapter, nothing)
end

# Mark NeuralPolicy as a leaf so fmap doesn't recurse into its fields.
# Our adapt_structure method below handles the actual device transfer.
MLDataDevices.isleaf(::NeuralPolicy) = true

function Adapt.adapt_structure(to::AbstractDevice, np::NeuralPolicy)
    new_params = to(np.params)
    new_states = to(np.states)
    return NeuralPolicy(
        np.model,
        new_params,
        new_states,
        np.action_space,
        np.adapter,
        nothing,
    )
end

"""
    extract_policy(model, ps, st, adapter) -> NeuralPolicy
    extract_policy(model, ps, st, adapter, norm_env::NormalizeWrapperEnv) -> NormWrapperPolicy

Create a lightweight deployment policy from a trained model and parameters.
"""
function extract_policy(model, ps, st, adapter::AbstractActionAdapter; action_space = nothing)
    as = action_space === nothing ? Models.action_space(model) : action_space
    return NeuralPolicy(model, ps, st, as, adapter, nothing)
end

function extract_policy(cache::RLCache)
    return extract_policy(
        cache.model,
        parameters(cache),
        states(cache),
        cache.adapter;
        action_space = Models.action_space(cache.model),
    )
end

function extract_policy(model, ps, st, adapter::AbstractActionAdapter, norm_env::NormalizeWrapperEnv)
    policy = extract_policy(model, ps, st, adapter)
    obs_rms = norm_env.obs_rms
    eps = norm_env.epsilon
    clip_obs = norm_env.clip_obs
    return NormWrapperPolicy(policy, obs_rms, eps, clip_obs)
end

function invalidate_cache!(np::NeuralPolicy)
    np.cache = nothing
    return np
end

# A single observation, a vector of observations, or a batched `(obs_dims..., n)` array
_is_single_obs(obs::AbstractVector{<:AbstractArray}, space) = false
_is_single_obs(obs, space) = size(obs) == size(space)

function _policy_obs_batch(obs, space)
    if obs isa AbstractVector{<:AbstractArray} || obs isa AbstractVector{<:Integer}
        return batch(obs, space)
    end
    return obs
end

function (np::NeuralPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = default_rng())
    space = Models.observation_space(np.model)
    single_obs = _is_single_obs(obs, space)
    obs_batch = single_obs ? reshape(obs, size(obs)..., 1) : _policy_obs_batch(obs, space)
    dev = current_device(np.params)
    obs_batch = obs_batch |> dev
    obs_batch = canonicalize_device_batch(dev, obs_batch)
    st = deployment_inference_state(np.states)
    actions_batched, _ = execute_deployment_predict_actions(
        dev,
        np,
        obs_batch,
        np.params,
        st;
        deterministic,
        rng,
    )
    actions_batched = cpu_device()(actions_batched)
    actions_vec = actions_batched isa AbstractVector ? collect(actions_batched) : collect(eachslice(actions_batched, dims = ndims(actions_batched)))
    env_actions = to_env.(Ref(np.adapter), actions_vec, Ref(np.action_space))
    if single_obs
        return env_actions[1]
    else
        return env_actions
    end
end

"""
    NormWrapperPolicy

Wraps a [`NeuralPolicy`](@ref) (or compatible policy) with observation normalization from a [`NormalizeWrapperEnv`](@ref), matching training-time obs scaling at deployment.
"""
struct NormWrapperPolicy{P <: AbstractPolicy, T <: AbstractFloat} <: AbstractPolicy
    policy::P
    obs_rms::RunningMeanStd{T}
    eps::T
    clip_obs::T
end

function (nwp::NormWrapperPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = default_rng())
    space = Models.observation_space(nwp.policy.model)
    single_obs = _is_single_obs(obs, space)
    # Normalize a copy; the caller's observations stay unchanged.
    obs_batch = single_obs ? reshape(copy(obs), size(obs)..., 1) : copy(_policy_obs_batch(obs, space))
    normalize_obs!(obs_batch, nwp.obs_rms, nwp.eps, nwp.clip_obs)
    actions = nwp.policy(obs_batch; deterministic, rng)
    return single_obs ? actions[1] : actions
end
