"""
    MultiAgentParallelEnv(envs::Vector{<:AbstractParallelEnv})

Concatenates several parallel environments into one, stepping them concurrently with
`@threads`. Observations, rewards and flags are stacked in the order of `envs`.
"""
struct MultiAgentParallelEnv{E <: AbstractParallelEnv} <: AbstractParallelEnv
    envs::Vector{E}
    env_counts::Vector{Int}  # Number of sub-envs in each parallel env
    total_envs::Int          # Sum of all env_counts

    function MultiAgentParallelEnv(envs::Vector{E}) where {E <: AbstractParallelEnv}
        @assert !isempty(envs) "Must provide at least one parallel environment"
        @assert all(env -> isequal(observation_space(env), observation_space(envs[1])), envs) "All sub-environments must have the same observation space"
        @assert all(env -> isequal(action_space(env), action_space(envs[1])), envs) "All sub-environments must have the same action space"
        env_counts = [number_of_envs(env) for env in envs]
        return new{E}(envs, env_counts, sum(env_counts))
    end
end

number_of_envs(env::MultiAgentParallelEnv) = env.total_envs
observation_space(env::MultiAgentParallelEnv) = observation_space(env.envs[1])
action_space(env::MultiAgentParallelEnv) = action_space(env.envs[1])

# Index range of each sub-env's envs in the stacked results
function _chunk_ranges(env::MultiAgentParallelEnv)
    stops = cumsum(env.env_counts)
    return [(stop - count + 1):stop for (stop, count) in zip(stops, env.env_counts)]
end

_stack_observations(parts) = cat(parts...; dims = ndims(first(parts)))

function observe(env::MultiAgentParallelEnv)
    return _stack_observations(observe.(env.envs))
end

function reset!(env::MultiAgentParallelEnv; seed::Union{Nothing, Integer} = nothing)
    ranges = _chunk_ranges(env)
    parts = Vector{Any}(undef, length(env.envs))
    @threads for k in eachindex(env.envs)
        parts[k] = reset!(env.envs[k]; seed = _sub_seed(seed, first(ranges[k]) - 1))
    end
    return _stack_observations(parts)
end

function step!(env::MultiAgentParallelEnv, actions::AbstractVector)
    n = number_of_envs(env)
    @assert length(actions) == n "Number of actions ($(length(actions))) must match total number of environments ($n)"
    ranges = _chunk_ranges(env)
    results = Vector{Any}(undef, length(env.envs))
    @threads for k in eachindex(env.envs)
        results[k] = step!(env.envs[k], view(actions, ranges[k]))
    end
    obs = _stack_observations([r[1] for r in results])
    rewards = reduce(vcat, [r[2] for r in results])
    terminateds = reduce(vcat, [r[3] for r in results])
    truncateds = reduce(vcat, [r[4] for r in results])
    final_obs = _stack_observations([r[5] for r in results])
    sub_infos = [r[6] for r in results]
    infos = if all(isnothing, sub_infos)
        nothing
    else
        reduce(vcat, [isnothing(i) ? fill(nothing, length(r)) : collect(Any, i) for (i, r) in zip(sub_infos, ranges)])
    end
    return obs, rewards, terminateds, truncateds, final_obs, infos
end

# MultiAgentParallelEnv show methods
function Base.show(io::IO, env::MultiAgentParallelEnv{E}) where {E}
    return print(io, "MultiAgentParallelEnv{", E, "}(", length(env.envs), " parallel envs, ", env.total_envs, " total envs)")
end

function Base.show(io::IO, ::MIME"text/plain", env::MultiAgentParallelEnv{E}) where {E}
    println(io, "MultiAgentParallelEnv{", E, "}")
    println(io, "  - Number of parallel environments: ", length(env.envs))
    println(io, "  - Total environments: ", env.total_envs)
    println(io, "  - Environment counts per parallel env: ", env.env_counts)
    println(io, "  - Observation space: ", observation_space(env))
    println(io, "  - Action space: ", action_space(env))
    for (i, sub_env) in enumerate(env.envs)
        println(io, "  - Parallel env ", i, " (", env.env_counts[i], " envs): ")
        show(io, sub_env)
        if i < length(env.envs)
            println(io)
        end
    end
    return
end
