"""
    MultiThreadedParallelEnv(envs::Vector)

Parallel environment that steps sub-environments concurrently with `@threads` (same
observation/action spaces, homogeneous env type).

Use for CPU-bound envs when parallel rollout helps; compare [`BroadcastedParallelEnv`](@ref).
"""
struct MultiThreadedParallelEnv{E <: AbstractEnv} <: AbstractParallelEnv
    envs::Vector{E}

    function MultiThreadedParallelEnv(envs::Vector{E}) where {E <: AbstractEnv}
        _check_homogeneous(envs)
        return new{E}(envs)
    end
end

function _foreach_env(f, penv::MultiThreadedParallelEnv)
    @threads for i in eachindex(penv.envs)
        f(i)
    end
    return nothing
end

# MultiThreadedParallelEnv show methods
function Base.show(io::IO, env::MultiThreadedParallelEnv{E}) where {E}
    return print(io, "MultiThreadedParallelEnv{", E, "}(", length(env.envs), " envs)")
end

function Base.show(io::IO, ::MIME"text/plain", env::MultiThreadedParallelEnv{E}) where {E}
    println(io, "MultiThreadedParallelEnv{", E, "}")
    println(io, "  - Number of environments: ", length(env.envs))
    println(io, "  - Observation space: ", observation_space(env))
    println(io, "  - Action space: ", action_space(env))
    print(io, "  environments: ")
    return show(io, env.envs[1])
end
