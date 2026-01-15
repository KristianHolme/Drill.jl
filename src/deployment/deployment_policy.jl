# Deployment-time policy (actor-only wrapper)

struct DeploymentPolicy{L, AD, S} <: AbstractPolicy
    layer::L
    params
    states::S
    action_space
    adapter::AD
end

"""
    extract_policy(agent) -> DeploymentPolicy

Create a lightweight deployment policy from a trained agent.
"""
function extract_policy(agent)
    layer = agent.layer
    ps = agent.train_state.parameters
    st = agent.train_state.states
    as = action_space(layer)
    adapter = agent.action_adapter
    return DeploymentPolicy(layer, ps, st, as, adapter)
end


function (dp::DeploymentPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())
    single_obs = false
    if obs in observation_space(dp.layer) #single observation, make into vector
        obs = [obs]
        single_obs = true
    end
    obs_batch = batch(obs, observation_space(dp.layer))
    actions, _ = predict_actions(dp.layer, obs_batch, dp.params, dp.states; deterministic = deterministic, rng = rng)
    env_actions = to_env.(Ref(dp.adapter), actions, Ref(dp.action_space))
    if single_obs
        return env_actions[1]
    else
        return env_actions
    end
end

#TODO: add tests
struct NormalizedDeploymentPolicy{P <: DeploymentPolicy, T <: AbstractFloat} <: AbstractPolicy
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
    return NormalizedDeploymentPolicy(policy, obs_rms, eps, clip_obs)
end

function (dp::NormalizedDeploymentPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())
    single_obs = false
    if obs in observation_space(dp.policy.layer) #single observation, make into vector
        single_obs = true
        obs = [obs]
    end
    normalize_obs!.(obs, Ref(dp.obs_rms), dp.eps, dp.clip_obs)
    actions = dp.policy(obs; deterministic = deterministic, rng = rng)
    if single_obs
        return actions[1]
    else
        return actions
    end
end

struct RandomPolicy{A <: AbstractSpace} <: AbstractPolicy
    action_space::A
end

function (rp::RandomPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())
    return rand(rng, rp.action_space)
end

function RandomPolicy(env::AbstractEnv)
    return RandomPolicy(action_space(env))
end

struct ZeroPolicy{V} <: AbstractPolicy
    action::V
end

function (zp::ZeroPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())
    return zp.action
end

function ZeroPolicy(env::AbstractEnv)
    action = rand(action_space(env))
    zero_action = action .* zero(eltype(action))
    return ZeroPolicy(zero_action)
end
