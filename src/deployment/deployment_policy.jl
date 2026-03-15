# Deployment-time policy (actor-only wrapper)

mutable struct NeuralPolicy{L, AD, S} <: AbstractPolicy
    layer::L
    params
    states::S
    action_space
    adapter::AD
    cache
end

function NeuralPolicy(layer::L, params, states::S, action_space, adapter::AD) where {L, AD, S}
    return NeuralPolicy(layer, params, states, action_space, adapter, nothing)
end

# Mark NeuralPolicy as a leaf so fmap doesn't recurse into its fields.
# Our adapt_structure method below handles the actual device transfer.
MLDataDevices.isleaf(::NeuralPolicy) = true

function Adapt.adapt_structure(to::MLDataDevices.AbstractDevice, np::NeuralPolicy)
    new_params = to(np.params)
    new_states = to(np.states)
    return NeuralPolicy(
        np.layer,
        new_params,
        new_states,
        np.action_space,
        np.adapter,
        nothing,
    )
end

"""
    extract_policy(agent) -> NeuralPolicy

Create a lightweight deployment policy from a trained agent.
"""
function extract_policy(agent)
    layer = agent.layer
    ps = agent.train_state.parameters
    st = agent.train_state.states
    as = action_space(layer)
    adapter = agent.action_adapter
    return NeuralPolicy(layer, ps, st, as, adapter, nothing)
end

function invalidate_cache!(np::NeuralPolicy)
    np.cache = nothing
    return np
end


function (np::NeuralPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())
    single_obs = false
    if !(obs isa AbstractVector{<:AbstractArray}) && size(obs) == size(observation_space(np.layer)) #single observation, make into vector
        single_obs = true
        obs_batch = reshape(obs, :, 1)
    else
        obs_batch = batch(obs, observation_space(np.layer))
    end
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
    actions_vec = actions_batched isa AbstractVector ? collect(actions_batched) : collect(eachslice(actions_batched, dims = ndims(actions_batched)))
    if dev isa MLDataDevices.ReactantDevice
        actions_vec = map(Array, actions_vec)
    end
    env_actions = to_env.(Ref(np.adapter), actions_vec, Ref(np.action_space))
    if single_obs
        return env_actions[1]
    else
        return env_actions
    end
end

#TODO: add tests
struct NormWrapperPolicy{P <: AbstractPolicy, T <: AbstractFloat} <: AbstractPolicy
    policy::P
    obs_rms::RunningMeanStd{T}
    eps::T
    clip_obs::T
end

function extract_policy(agent, norm_env::NormalizeWrapperEnv)
    policy = extract_policy(agent)
    obs_rms = norm_env.obs_rms
    eps = norm_env.epsilon
    clip_obs = norm_env.clip_obs
    return NormWrapperPolicy(policy, obs_rms, eps, clip_obs)
end

function (nwp::NormWrapperPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())
    single_obs = false
    if size(obs) == size(observation_space(nwp.policy.layer)) #single observation, make into vector
        single_obs = true
        obs = [obs]
    end
    normalize_obs!.(obs, Ref(nwp.obs_rms), nwp.eps, nwp.clip_obs)
    actions = nwp.policy(obs; deterministic, rng)
    if single_obs
        return actions[1]
    else
        return actions
    end
end

"""
    RandomPolicy(env)
    RandomPolicy(action_space)

A policy that returns a random action from the action space.

# Examples
```julia
using ClassicControlEnvironments
env = CartPoleEnv()
policy = RandomPolicy(env)
action = policy(obs; deterministic = true, rng = Random.Xoshiro(123))
```
"""
struct RandomPolicy{A <: AbstractSpace} <: AbstractPolicy
    action_space::A
end

function (rp::RandomPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())
    return rand(rng, rp.action_space)
end

function RandomPolicy(env::AbstractEnv)
    return RandomPolicy(action_space(env))
end


"""
    ConstantPolicy(action)

A policy that returns a constant action. Will throw an error if deterministic is false.
 Will warn if rng is not nothing. Will not use the rng.

# Examples
```julia
using ClassicControlEnvironments
env = CartPoleEnv()
policy = ConstantPolicy([0.0f0])
action = policy(obs; deterministic = true, rng = Random.Xoshiro(123))
```
"""
struct ConstantPolicy{A} <: AbstractPolicy
    action::A
end

function (cp::ConstantPolicy)(obs; deterministic::Bool = true, rng::Union{Nothing, AbstractRNG} = nothing)
    !deterministic && error("ConstantPolicy is deterministic")
    !isnothing(rng) && warn("rng is not used by ConstantPolicy")
    return cp.action
end
