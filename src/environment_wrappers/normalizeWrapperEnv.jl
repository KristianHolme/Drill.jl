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
Drill.unwrap(env::NormalizeWrapperEnv) = env.env

# Convenience constructor
function NormalizeWrapperEnv(env::E; kwargs...) where {E <: AbstractParallelEnv}
    return NormalizeWrapperEnv{E, Float32}(env; kwargs...)
end

# Forward basic properties
observation_space(env::NormalizeWrapperEnv) = observation_space(env.env)
action_space(env::NormalizeWrapperEnv) = action_space(env.env)
number_of_envs(env::NormalizeWrapperEnv) = number_of_envs(env.env)

function reset!(env::NormalizeWrapperEnv{E, T}) where {E, T}
    reset!(env.env)
    obs = observe(env.env)

    # Store original observations BEFORE normalization
    #should we also store rewards or something?
    env.old_obs .= batch(obs, observation_space(env))
    env.returns .= zero(T)

    return nothing
end

function observe(env::NormalizeWrapperEnv{E, T}) where {E, T}
    obs = observe(env.env)

    # Store original observations and rewards for access
    env.old_obs .= batch(obs, observation_space(env))

    # Update observation statistics if in training mode
    if env.training && env.norm_obs
        obs_batch = batch(obs, observation_space(env))
        update!(env.obs_rms, obs_batch)
    end
    #FIXME: type instability here?
    normalize_obs!.(obs, Ref(env))
    return obs
end

function act!(env::NormalizeWrapperEnv{E, T}, actions::AbstractVector) where {E, T}
    rewards, terminateds, truncateds, infos = act!(env.env, actions)
    env.old_rewards .= rewards

    # Update reward statistics and normalize
    if env.training && env.norm_reward
        update_reward_stats!(env, rewards)
    end
    normalize_rewards!(rewards, env)

    # Reset returns for terminated environments
    dones = terminateds .| truncateds
    for i in findall(dones)
        env.returns[i] = zero(T)
    end

    # Normalize terminal observations in infos
    for i in findall(truncateds)
        if haskey(infos[i], "terminal_observation")
            term_obs = infos[i]["terminal_observation"]
            normalize_obs!(term_obs, env)
            infos[i]["terminal_observation"] = term_obs
        end
    end

    return rewards, terminateds, truncateds, infos
end

function update_reward_stats!(env::NormalizeWrapperEnv, rewards::Vector{T}) where {T <: AbstractFloat}
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

# Get original (unnormalized) observations and rewards
get_original_obs(env::NormalizeWrapperEnv) = eachslice(env.old_obs, dims = ndims(env.old_obs))
get_original_rewards(env::NormalizeWrapperEnv) = env.old_rewards

# Forward other methods
terminated(env::NormalizeWrapperEnv) = terminated(env.env)
truncated(env::NormalizeWrapperEnv) = truncated(env.env)
function get_info(env::NormalizeWrapperEnv)
    infos = get_info(env.env)
    terminateds = terminated(env.env)
    truncateds = truncated(env.env)
    dones = terminateds .| truncateds

    for i in findall(dones)
        if haskey(infos[i], "terminal_observation")
            term_obs = infos[i]["terminal_observation"]
            normalize_obs!(term_obs, env)
            infos[i]["terminal_observation"] = term_obs
        end
    end
    return infos
end

function Random.seed!(env::NormalizeWrapperEnv, seed::Integer)
    Random.seed!(env.env, seed)
    return env
end

# Training mode control
set_training(env::AbstractEnv, ::Bool) = env #default to no-op
#TODO: fix/doc this
is_training(env::AbstractEnv) = true
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

#syncs the eval env stats to be same as training env
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
