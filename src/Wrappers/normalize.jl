# Running mean and standard deviation tracker for normalization
"""
    RunningMeanStd{T}

Tracks running mean and standard deviation using Welford's online algorithm.
Similar to stable-baselines3's RunningMeanStd.
"""
mutable struct RunningMeanStd{T <: AbstractFloat}
    mean::Array{T}
    var::Array{T}
    count::Int

    function RunningMeanStd{T}(shape::Tuple{Vararg{Int}}) where {T <: AbstractFloat}
        return new{T}(zeros(T, shape), ones(T, shape), 0)
    end
end

RunningMeanStd(shape::Tuple{Vararg{Int}}) = RunningMeanStd{Float32}(shape)
RunningMeanStd(::Type{T}, shape::Tuple{Vararg{Int}}) where {T <: AbstractFloat} = RunningMeanStd{T}(shape)

function update!(rms::RunningMeanStd{T}, batch::AbstractArray{T}) where {T}
    batch_mean = mean(batch, dims = ndims(batch))
    batch_var = var(batch, dims = ndims(batch), corrected = false)
    batch_count = size(batch, ndims(batch))
    return update_from_moments!(rms, batch_mean, batch_var, batch_count)
end

function update_from_moments!(
        rms::RunningMeanStd{T}, batch_mean::AbstractArray{T},
        batch_var::AbstractArray{T}, batch_count::Int
    ) where {T}
    return if rms.count == 0
        rms.mean .= dropdims(batch_mean, dims = ndims(batch_mean))
        rms.var .= dropdims(batch_var, dims = ndims(batch_var))
        rms.count = batch_count
    else
        delta = dropdims(batch_mean, dims = ndims(batch_mean)) .- rms.mean
        total_count = rms.count + batch_count

        new_mean = rms.mean .+ delta .* batch_count ./ total_count
        m_a = rms.var .* rms.count
        m_b = dropdims(batch_var, dims = ndims(batch_var)) .* batch_count
        M2 = m_a .+ m_b .+ delta .^ 2 .* rms.count .* batch_count ./ total_count
        new_var = M2 ./ total_count

        rms.mean .= new_mean
        rms.var .= new_var
        rms.count = total_count
    end
end

"""
    NormalizeWrapperEnv

[`AbstractParallelEnvWrapper`](@ref) that optionally normalizes observations and/or rewards using running statistics ([`RunningMeanStd`](@ref)), with clipping. Used in training to stabilize value learning.

Toggle training vs inference behavior with [`set_training`](@ref) / [`is_training`](@ref); sync stats across parallel copies with [`sync_normalization_stats!`](@ref) when needed.
"""
struct NormalizeWrapperEnv{E <: AbstractParallelEnv, T <: AbstractFloat} <: AbstractParallelEnvWrapper{E}
    env::E
    obs_rms::RunningMeanStd{T}
    ret_rms::RunningMeanStd{T}
    returns::Vector{T}

    # Configuration
    training::Bool
    norm_obs::Bool
    norm_reward::Bool
    clip_obs::T
    clip_reward::T
    gamma::T
    epsilon::T

    # Cache for original observations/rewards
    old_obs::Array{T}
    old_rewards::Vector{T}
end
function NormalizeWrapperEnv{E, T}(
        env::E;
        training::Bool = true,
        norm_obs::Bool = true,
        norm_reward::Bool = true,
        clip_obs::T = T(10.0),
        clip_reward::T = T(10.0),
        gamma::T = T(0.99),
        epsilon::T = T(1.0e-8)
    ) where {E <: AbstractParallelEnv, T <: AbstractFloat}

    obs_space = observation_space(env)
    n_envs = number_of_envs(env)

    # Initialize running statistics
    obs_rms = RunningMeanStd(T, size(obs_space))
    ret_rms = RunningMeanStd(T, ())
    returns = zeros(T, n_envs)

    # Initialize cache arrays
    old_obs = Array{T}(undef, size(obs_space)..., n_envs)
    old_rewards = Vector{T}(undef, n_envs)

    return NormalizeWrapperEnv{E, T}(
        env, obs_rms, ret_rms, returns, training, norm_obs, norm_reward,
        clip_obs, clip_reward, gamma, epsilon, old_obs, old_rewards
    )
end
DrillInterface.unwrap(env::NormalizeWrapperEnv) = env.env

# Convenience constructor
function NormalizeWrapperEnv(env::E; kwargs...) where {E <: AbstractParallelEnv}
    return NormalizeWrapperEnv{E, Float32}(env; kwargs...)
end

# Forward basic properties
observation_space(env::NormalizeWrapperEnv) = observation_space(env.env)
action_space(env::NormalizeWrapperEnv) = action_space(env.env)
number_of_envs(env::NormalizeWrapperEnv) = number_of_envs(env.env)

function reset!(env::NormalizeWrapperEnv{E, T}; seed::Union{Nothing, Integer} = nothing) where {E, T}
    obs = reset!(env.env; seed)
    env.returns .= zero(T)
    return _normalize_new_obs!(env, obs)
end

# Record and normalize a batched observation that just came from the inner env. The
# result is a new array: the inner env may still own `obs`.
function _normalize_new_obs!(env::NormalizeWrapperEnv, obs::AbstractArray)
    env.old_obs .= obs
    if env.training && env.norm_obs
        update!(env.obs_rms, env.old_obs)
    end
    return _normalized_copy(obs, env)
end

function _normalized_copy(obs::AbstractArray, env::NormalizeWrapperEnv)
    normalized = copy(obs)
    normalize_obs!(normalized, env)
    return normalized
end

"""
    observe(env::NormalizeWrapperEnv)

The normalized current observation. Unlike `reset!` and `step!`, this does not update the
running statistics.
"""
function observe(env::NormalizeWrapperEnv)
    return _normalized_copy(observe(env.env), env)
end

function step!(env::NormalizeWrapperEnv{E, T}, actions::AbstractVector) where {E, T}
    obs, rewards, terminateds, truncateds, final_obs, infos = step!(env.env, actions)
    env.old_rewards .= rewards

    if env.training && env.norm_reward
        update_reward_stats!(env, rewards)
    end
    norm_rewards = copy(rewards)
    normalize_rewards!(norm_rewards, env)

    for i in eachindex(terminateds, truncateds)
        if terminateds[i] || truncateds[i]
            env.returns[i] = zero(T)
        end
    end

    norm_obs = _normalize_new_obs!(env, obs)
    norm_final_obs = _normalized_copy(final_obs, env)
    return norm_obs, norm_rewards, terminateds, truncateds, norm_final_obs, infos
end

function update_reward_stats!(env::NormalizeWrapperEnv, rewards::AbstractVector{T}) where {T <: AbstractFloat}
    env.returns .= env.returns .* env.gamma .+ rewards
    # Update return statistics (single value, so we reshape for consistency)
    return update!(env.ret_rms, reshape(env.returns, 1, length(env.returns)))
end


function normalize_obs!(obs, obs_rms::RunningMeanStd, epsilon::T, clip_obs::T) where {T <: AbstractFloat}
    # Normalize using running statistics
    @. obs = (obs .- obs_rms.mean) ./ sqrt.(obs_rms.var .+ epsilon)
    clamp!(obs, -clip_obs, clip_obs)
    return nothing
end

function normalize_obs!(obs, env::NormalizeWrapperEnv)
    if !env.norm_obs
        return obs
    end
    return normalize_obs!(obs, env.obs_rms, env.epsilon, env.clip_obs)
end

function normalize_rewards!(rewards, env::NormalizeWrapperEnv)
    if !env.norm_reward
        return rewards
    end

    # Normalize rewards using return statistics
    @. rewards = rewards ./ sqrt(env.ret_rms.var[1] + env.epsilon)
    clamp!(rewards, -env.clip_reward, env.clip_reward)
    return nothing
end

#TODO: should these methods not return nothing?
function unnormalize_obs!(obs, obs_rms::RunningMeanStd, epsilon::T) where {T <: AbstractFloat}
    @. obs = obs .* sqrt.(obs_rms.var .+ epsilon) .+ obs_rms.mean
    return nothing
end
function unnormalize_obs!(obs, env::NormalizeWrapperEnv)
    if !env.norm_obs
        return obs
    end
    unnormalize_obs!(obs, env.obs_rms, env.epsilon)
    return nothing
end

function unnormalize_rewards!(rewards, env::NormalizeWrapperEnv)
    if !env.norm_reward
        return rewards
    end
    @. rewards = rewards .* sqrt(env.ret_rms.var[1] + env.epsilon)
    return nothing
end

"""
    get_original_obs(env::NormalizeWrapperEnv) -> Array

A copy of the last batched observation from the inner env, before normalization.
"""
get_original_obs(env::NormalizeWrapperEnv) = copy(env.old_obs)

"""
    get_original_rewards(env::NormalizeWrapperEnv) -> Vector

A copy of the last rewards from the inner env, before normalization.
"""
get_original_rewards(env::NormalizeWrapperEnv) = copy(env.old_rewards)

# Training mode control
"""
    set_training(env, training::Bool)

Return an environment with training mode set when applicable (e.g. [`NormalizeWrapperEnv`](@ref)); default no-op for other envs.
"""
set_training(env::Union{AbstractEnv, AbstractParallelEnv}, ::Bool) = env #default to no-op

"""
    is_training(env) -> Bool

Whether `env` is in training mode (obs/reward normalization updates when wrapped with [`NormalizeWrapperEnv`](@ref)); default `true` for other envs.
"""
is_training(env::Union{AbstractEnv, AbstractParallelEnv}) = true
set_training(env::NormalizeWrapperEnv{E, T}, training::Bool) where {E, T} = @set env.training = training
is_training(env::NormalizeWrapperEnv{E, T}) where {E, T} = env.training

# Save/load functionality for normalization statistics
"""
    save_normalization_stats(env::NormalizeWrapperEnv, filepath::String)

Save the normalization statistics (running mean/std) to a file using JLD2.
"""
function save_normalization_stats(env::NormalizeWrapperEnv, filepath::String)
    return save(
        filepath, Dict(
            "obs_mean" => env.obs_rms.mean,
            "obs_var" => env.obs_rms.var,
            "obs_count" => env.obs_rms.count,
            "ret_mean" => env.ret_rms.mean,
            "ret_var" => env.ret_rms.var,
            "ret_count" => env.ret_rms.count,
            "clip_obs" => env.clip_obs,
            "clip_reward" => env.clip_reward,
            "gamma" => env.gamma,
            "epsilon" => env.epsilon
        )
    )
end

"""
    load_normalization_stats!(env::NormalizeWrapperEnv, filepath::String)

Load normalization statistics from a file into the environment using JLD2.
"""
function load_normalization_stats!(env::NormalizeWrapperEnv{E, T}, filepath::String) where {E, T <: AbstractFloat}
    stats = load(filepath)

    # Load observation statistics
    env.obs_rms.mean .= T.(stats["obs_mean"])
    env.obs_rms.var .= T.(stats["obs_var"])
    env.obs_rms.count = stats["obs_count"]

    # Load return statistics
    env.ret_rms.mean .= T.(stats["ret_mean"])
    env.ret_rms.var .= T.(stats["ret_var"])
    env.ret_rms.count = stats["ret_count"]

    return env
end

"""
    sync_normalization_stats!(eval_env::NormalizeWrapperEnv, train_env::NormalizeWrapperEnv)

Copy running normalization statistics from `train_env` to `eval_env` so evaluation uses the same obs/reward scaling.
"""
function sync_normalization_stats!(eval_env::NormalizeWrapperEnv{E1, T}, train_env::NormalizeWrapperEnv{E2, T}) where {E1, E2, T}
    eval_env.obs_rms.mean .= train_env.obs_rms.mean
    eval_env.obs_rms.var .= train_env.obs_rms.var
    eval_env.obs_rms.count = train_env.obs_rms.count
    eval_env.ret_rms.mean .= train_env.ret_rms.mean
    eval_env.ret_rms.var .= train_env.ret_rms.var
    eval_env.ret_rms.count = train_env.ret_rms.count
    eval_env.returns .= zero(T) #reset returns. We allow n_envs to be different for the two envs, so we dont sync the current returns #reset returns. We allow n_envs to be different for the two envs, so we dont sync the current returns
    return nothing
end

# NormalizeWrapperEnv show methods
function Base.show(io::IO, env::NormalizeWrapperEnv{E, T}) where {E, T}
    return print(io, "NormalizeWrapperEnv{", E, ",", T, "}(", number_of_envs(env), " envs)")
end

function Base.show(io::IO, ::MIME"text/plain", env::NormalizeWrapperEnv{E, T}) where {E, T}
    println(io, "NormalizeWrapperEnv{", E, ",", T, "}")
    println(io, "  - Training mode: ", env.training)
    println(io, "  - Normalize observations: ", env.norm_obs)
    println(io, "  - Normalize rewards: ", env.norm_reward)
    println(io, "  - Observation clip: ±", env.clip_obs)
    println(io, "  - Reward clip: ±", env.clip_reward)
    println(io, "  - Discount factor (γ): ", env.gamma)
    println(io, "  - Epsilon: ", env.epsilon)

    if env.obs_rms.count > 0
        println(io, "  - Observation stats (n=", env.obs_rms.count, "):")
        println(io, "    • Mean: ", round.(env.obs_rms.mean, digits = 3))
        println(io, "    • Std: ", round.(sqrt.(env.obs_rms.var .+ env.epsilon), digits = 3))
    else
        println(io, "  - Observation stats: Not initialized")
    end

    if env.ret_rms.count > 0
        println(io, "  - Return stats (n=", env.ret_rms.count, "):")
        println(io, "    • Std: ", round(sqrt(env.ret_rms.var[1] + env.epsilon), digits = 3))
    else
        println(io, "  - Return stats: Not initialized")
    end

    print(io, "  wrapped environment: ")
    return show(io, env.env)
end
