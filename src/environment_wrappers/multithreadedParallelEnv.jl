"""
    MultiThreadedParallelEnv(envs::Vector)

Parallel environment that steps sub-environments concurrently with `@threads` (same observation/action spaces, homogeneous env type).

Use for CPU-bound envs when parallel rollout helps; compare [`BroadcastedParallelEnv`](@ref).
"""
struct MultiThreadedParallelEnv{E <: AbstractEnv} <: AbstractParallelEnv
    envs::Vector{E}
    function MultiThreadedParallelEnv(envs::Vector{E}) where {E <: AbstractEnv}
        @assert all(env -> typeof(env) == E, envs) "All environments must be of the same type"
        @assert all(env -> isequal(observation_space(env), observation_space(envs[1])), envs) "All environments must have the same observation space"
        @assert all(env -> isequal(action_space(env), action_space(envs[1])), envs) "All environments must have the same action space"
        return new{E}(envs)
    end
end

function reset!(env::MultiThreadedParallelEnv{E}) where {E <: AbstractEnv}
    @threads for sub_env in env.envs
        reset!(sub_env)
    end
    return nothing
end

#TODO: check if this typing is correct
function observe(env::MultiThreadedParallelEnv{E}) where {E <: AbstractEnv}
    observations = Vector{Array{eltype(observation_space(env)), length(size(observation_space(env)))}}(undef, length(env.envs))
    @threads for i in 1:length(env.envs)
        observations[i] = observe(env.envs[i])
    end
    return observations
end

function terminated(env::MultiThreadedParallelEnv{E}) where {E <: AbstractEnv}
    return terminated.(env.envs)
end

function truncated(env::MultiThreadedParallelEnv{E}) where {E <: AbstractEnv}
    return truncated.(env.envs)
end

function action_space(env::MultiThreadedParallelEnv{E}) where {E <: AbstractEnv}
    return action_space(env.envs[1])
end

function observation_space(env::MultiThreadedParallelEnv{E}) where {E <: AbstractEnv}
    return observation_space(env.envs[1])
end

function get_info(env::MultiThreadedParallelEnv{E}) where {E <: AbstractEnv}
    return get_info.(env.envs)
end

function act!(env::MultiThreadedParallelEnv{E}, actions::AbstractVector) where {E <: AbstractEnv}
    @assert length(actions) == length(env.envs) "Number of actions ($(length(actions))) must match number of environments ($(length(env.envs)))"

    T = eltype(observation_space(env))
    rewards = Vector{T}(undef, length(env.envs))
    terminateds = Vector{Bool}(undef, length(env.envs))
    truncateds = Vector{Bool}(undef, length(env.envs))
    infos = Vector{Dict{String, Any}}(undef, length(env.envs))

    @threads for i in 1:length(env.envs)
        rewards[i] = act!(env.envs[i], actions[i])

        # Capture termination status BEFORE reset
        terminateds[i] = terminated(env.envs[i])
        truncateds[i] = truncated(env.envs[i])
        infos[i] = get_info(env.envs[i])

        if truncateds[i]
            infos[i]["terminal_observation"] = observe(env.envs[i])
        end

        if terminateds[i] || truncateds[i]
            reset!(env.envs[i])
        end
    end

    return rewards, terminateds, truncateds, infos
end

number_of_envs(env::MultiThreadedParallelEnv) = length(env.envs)

# MultiThreadedParallelEnv show methods
function Base.show(io::IO, env::MultiThreadedParallelEnv{E}) where {E}
    return print(io, "MultiThreadedParallelEnv{", E, "}(", length(env.envs), " envs)")
end

function Base.show(io::IO, ::MIME"text/plain", env::MultiThreadedParallelEnv{E}) where {E}
    println(io, "MultiThreadedParallelEnv{", E, "}")
    println(io, "  - Number of environments: ", length(env.envs))
    obs_space = observation_space(env)
    act_space = action_space(env)
    println(io, "  - Observation space: ", obs_space)
    println(io, "  - Action space: ", act_space)
    print(io, "  environments: ")
    return show(io, env.envs[1])
end
