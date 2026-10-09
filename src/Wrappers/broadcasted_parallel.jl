"""
    BroadcastedParallelEnv(envs::Vector)

Parallel environment that steps `envs` one after another on the calling thread (same
spaces, homogeneous type).

Prefer when threading overhead dominates or env stepping is already cheap.
"""
struct BroadcastedParallelEnv{E <: AbstractEnv} <: AbstractParallelEnv
    envs::Vector{E}

    function BroadcastedParallelEnv(envs::Vector{E}) where {E <: AbstractEnv}
        _check_homogeneous(envs)
        return new{E}(envs)
    end
end

_foreach_env(f, penv::BroadcastedParallelEnv) = foreach(f, eachindex(penv.envs))

# BroadcastedParallelEnv show methods
function Base.show(io::IO, env::BroadcastedParallelEnv{E}) where {E}
    return print(io, "BroadcastedParallelEnv{", E, "}(", length(env.envs), " envs)")
end

function Base.show(io::IO, ::MIME"text/plain", env::BroadcastedParallelEnv{E}) where {E}
    println(io, "BroadcastedParallelEnv{", E, "}")
    println(io, "  - Number of environments: ", length(env.envs))
    println(io, "  - Observation space: ", observation_space(env))
    println(io, "  - Action space: ", action_space(env))
    print(io, "  environments: ")
    return show(io, env.envs[1])
end
