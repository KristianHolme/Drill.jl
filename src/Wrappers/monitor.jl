"""
    EpisodeStats{T}(stats_window)

Rolling buffers of recent finished-episode returns and lengths (used by [`MonitorWrapperEnv`](@ref)).
"""
struct EpisodeStats{T <: AbstractFloat}
    episode_returns::CircularBuffer{T}
    episode_lengths::CircularBuffer{Int}
end
function EpisodeStats{T}(stats_window::Int) where {T}
    return EpisodeStats{T}(CircularBuffer{T}(stats_window), CircularBuffer{Int}(stats_window))
end

"""
    MonitorWrapperEnv(env, stats_window=100)

Wraps a parallel environment to track per-env episode returns and lengths. Finished
episodes go into rolling buffers ([`EpisodeStats`](@ref)) for logging, and the last
finished episode of each env is kept in `last_episode_returns` / `last_episode_lengths`
for [`evaluate`](@ref).

Use when you want stable episode metrics under vectorized resets.
"""
struct MonitorWrapperEnv{E <: AbstractParallelEnv, T <: AbstractFloat} <: AbstractParallelEnvWrapper{E}
    env::E
    current_episode_lengths::Vector{Int}
    current_episode_returns::Vector{T}
    last_episode_lengths::Vector{Int}
    last_episode_returns::Vector{T}
    episode_stats::EpisodeStats{T}
end

function MonitorWrapperEnv(env::E, stats_window::Int = 100) where {E <: AbstractParallelEnv}
    T = reward_type(observation_space(env))
    n = number_of_envs(env)
    return MonitorWrapperEnv{E, T}(
        env,
        zeros(Int, n),
        zeros(T, n),
        zeros(Int, n),
        zeros(T, n),
        EpisodeStats{T}(stats_window),
    )
end

observe(monitor_env::MonitorWrapperEnv) = observe(monitor_env.env)
action_space(monitor_env::MonitorWrapperEnv) = action_space(monitor_env.env)
observation_space(monitor_env::MonitorWrapperEnv) = observation_space(monitor_env.env)
number_of_envs(monitor_env::MonitorWrapperEnv) = number_of_envs(monitor_env.env)

function reset!(monitor_env::MonitorWrapperEnv; seed::Union{Nothing, Integer} = nothing)
    obs = reset!(monitor_env.env; seed)
    # Episodes cut short by a reset do not count towards the stats
    monitor_env.current_episode_lengths .= 0
    monitor_env.current_episode_returns .= 0
    return obs
end

function step!(monitor_env::MonitorWrapperEnv, actions::AbstractVector)
    obs, rewards, terminateds, truncateds, final_obs, infos = step!(monitor_env.env, actions)

    monitor_env.current_episode_returns .+= rewards
    monitor_env.current_episode_lengths .+= 1

    for i in eachindex(terminateds, truncateds)
        if terminateds[i] || truncateds[i]
            ret = monitor_env.current_episode_returns[i]
            len = monitor_env.current_episode_lengths[i]
            push!(monitor_env.episode_stats.episode_returns, ret)
            push!(monitor_env.episode_stats.episode_lengths, len)
            monitor_env.last_episode_returns[i] = ret
            monitor_env.last_episode_lengths[i] = len
            monitor_env.current_episode_returns[i] = 0
            monitor_env.current_episode_lengths[i] = 0
        end
    end

    return obs, rewards, terminateds, truncateds, final_obs, infos
end

unwrap(env::MonitorWrapperEnv) = env.env

function log_stats(env::MonitorWrapperEnv{E, T}, logger::AbstractTrainingLogger) where {E, T}
    if length(env.episode_stats.episode_returns) > 0
        log_scalar!(logger, "env/ep_rew_mean", mean(env.episode_stats.episode_returns))
        log_scalar!(logger, "env/ep_len_mean", mean(env.episode_stats.episode_lengths))
    end
    return nothing
end

function log_stats(env::AbstractParallelEnvWrapper, logger::AbstractTrainingLogger)
    return log_stats(unwrap(env), logger)
end


# MonitorWrapperEnv show methods
function Base.show(io::IO, env::MonitorWrapperEnv{E, T}) where {E, T}
    return print(io, "MonitorWrapperEnv{", E, ",", T, "}(", number_of_envs(env), " envs)")
end

function Base.show(io::IO, ::MIME"text/plain", env::MonitorWrapperEnv{E, T}) where {E, T}
    println(io, "MonitorWrapperEnv{", E, ",", T, "}")
    println(io, "  - Number of environments: ", number_of_envs(env))
    println(io, "  - Stats window size: ", env.episode_stats.episode_returns.capacity)

    if length(env.episode_stats.episode_returns) > 0
        println(io, "  - Episode statistics (", length(env.episode_stats.episode_returns), " episodes):")
        println(io, "    • Mean return: ", round(mean(env.episode_stats.episode_returns), digits = 3))
        println(io, "    • Mean length: ", round(mean(env.episode_stats.episode_lengths), digits = 1))
        println(
            io, "    • Return range: [", round(minimum(env.episode_stats.episode_returns), digits = 3),
            ", ", round(maximum(env.episode_stats.episode_returns), digits = 3), "]"
        )
    else
        println(io, "  - Episode statistics: No completed episodes")
    end

    # Show current episode progress
    any_active = any(x -> x > 0, env.current_episode_lengths)
    if any_active
        active_envs = sum(x -> x > 0, env.current_episode_lengths)
        max_len = maximum(env.current_episode_lengths)
        max_ret = maximum(env.current_episode_returns)
        println(
            io, "  - Current episodes: ", active_envs, " active, max length: ", max_len,
            ", max return: ", round(max_ret, digits = 3)
        )
    else
        println(io, "  - Current episodes: None active")
    end

    print(io, "  wrapped environment: ")
    return show(io, env.env)
end
