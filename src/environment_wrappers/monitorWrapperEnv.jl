struct EpisodeStats{T <: AbstractFloat}
    episode_returns::CircularBuffer{T}
    episode_lengths::CircularBuffer{Int}
end
function EpisodeStats{T}(stats_window::Int) where {T}
    return EpisodeStats{T}(CircularBuffer{T}(stats_window), CircularBuffer{Int}(stats_window))
end

struct MonitorWrapperEnv{E <: AbstractParallelEnv, T} <: AbstractParallelEnvWrapper{E} where {T <: AbstractFloat}
    env::E
    current_episode_lengths::Vector{Int}
    current_episode_returns::Vector{T}
    episode_stats::EpisodeStats{T}
end

function MonitorWrapperEnv(env::E, stats_window::Int = 100) where {E <: AbstractParallelEnv}
    T = eltype(observation_space(env))
    return MonitorWrapperEnv{E, T}(
        env,
        zeros(Int, number_of_envs(env)),
        zeros(T, number_of_envs(env)),
        EpisodeStats{T}(stats_window)
    )
end

#TODO clean up this so its not necessary to forward all the methods
observe(monitor_env::MonitorWrapperEnv{E, T}) where {E, T} = observe(monitor_env.env)
terminated(monitor_env::MonitorWrapperEnv{E, T}) where {E, T} = terminated(monitor_env.env)
truncated(monitor_env::MonitorWrapperEnv{E, T}) where {E, T} = truncated(monitor_env.env)
get_info(monitor_env::MonitorWrapperEnv{E, T}) where {E, T} = get_info(monitor_env.env)
action_space(monitor_env::MonitorWrapperEnv{E, T}) where {E, T} = action_space(monitor_env.env)
observation_space(monitor_env::MonitorWrapperEnv{E, T}) where {E, T} = observation_space(monitor_env.env)
number_of_envs(monitor_env::MonitorWrapperEnv{E, T}) where {E, T} = number_of_envs(monitor_env.env)
Random.seed!(monitor_env::MonitorWrapperEnv{E, T}, seed::Integer) where {E, T} = Random.seed!(monitor_env.env, seed)

function reset!(monitor_env::MonitorWrapperEnv{E, T}) where {E, T}
    Drill.reset!(monitor_env.env)
    #dont count the current episodes to the stats, since they are manually stopped
    monitor_env.current_episode_lengths .= 0
    monitor_env.current_episode_returns .= 0
    return nothing
end

function act!(monitor_env::MonitorWrapperEnv{E, T}, actions::AbstractVector) where {E, T}
    rewards, terminateds, truncateds, infos = act!(monitor_env.env, actions)

    monitor_env.current_episode_returns .+= rewards
    monitor_env.current_episode_lengths .+= 1
    dones = terminateds .| truncateds

    for i in findall(dones)
        push!(monitor_env.episode_stats.episode_returns, monitor_env.current_episode_returns[i])
        push!(monitor_env.episode_stats.episode_lengths, monitor_env.current_episode_lengths[i])
        infos[i]["episode"] = Dict("r" => monitor_env.current_episode_returns[i], "l" => monitor_env.current_episode_lengths[i])
        monitor_env.current_episode_returns[i] = 0
        monitor_env.current_episode_lengths[i] = 0
    end

    return rewards, terminateds, truncateds, infos
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
