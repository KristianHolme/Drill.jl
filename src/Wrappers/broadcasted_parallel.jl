"""
    BroadcastedParallelEnv(envs::Vector)

Vectorized parallel environment: `act!`, `observe`, etc. broadcast over `envs` on a single thread (same spaces, homogeneous type).

Prefer when threading overhead dominates or env stepping is already cheap.
"""
struct BroadcastedParallelEnv{E <: AbstractEnv} <: AbstractParallelEnv
    envs::Vector{E}

    function BroadcastedParallelEnv(envs::Vector{E}) where {E <: AbstractEnv}
        @assert all(env -> typeof(env) == E, envs) "All environments must be of the same type"
        @assert all(env -> isequal(observation_space(env), observation_space(envs[1])), envs) "All environments must have the same observation space"
        @assert all(env -> isequal(action_space(env), action_space(envs[1])), envs) "All environments must have the same action space"
        return new{E}(envs)
    end
end

function reset!(env::BroadcastedParallelEnv{E}) where {E <: AbstractEnv}
    reset!.(env.envs)
    return nothing
end

function observe(env::BroadcastedParallelEnv{E}) where {E <: AbstractEnv}
    return observe.(env.envs)
end

function terminated(env::BroadcastedParallelEnv{E}) where {E <: AbstractEnv}
    return terminated.(env.envs)
end

function truncated(env::BroadcastedParallelEnv{E}) where {E <: AbstractEnv}
    return truncated.(env.envs)
end

function action_space(env::BroadcastedParallelEnv{E}) where {E <: AbstractEnv}
    return action_space(env.envs[1])
end

function observation_space(env::BroadcastedParallelEnv{E}) where {E <: AbstractEnv}
    return observation_space(env.envs[1])
end

function get_info(env::BroadcastedParallelEnv{E}) where {E <: AbstractEnv}
    return get_info.(env.envs)
end

function act!(env::BroadcastedParallelEnv{E}, actions::AbstractVector) where {E <: AbstractEnv}
    @assert length(actions) == length(env.envs) "Number of actions ($(length(actions))) must match number of environments ($(length(env.envs)))"

    # Use broadcasting for the main operations
    rewards = act!.(env.envs, actions)

    # Capture termination status BEFORE reset
    terminateds = terminated.(env.envs)
    truncateds = truncated.(env.envs)

    # Update infos using broadcasting
    infos = get_info.(env.envs)

    # Handle terminal observations for truncated environments
    for i in 1:length(env.envs)
        if truncateds[i]
            infos[i]["terminal_observation"] = observe(env.envs[i])
        end

        if terminateds[i] || truncateds[i]
            reset!(env.envs[i])
        end
    end

    return rewards, terminateds, truncateds, infos
end

number_of_envs(env::BroadcastedParallelEnv) = length(env.envs)
unwrap_all(env::BroadcastedParallelEnv) = env.envs


# BroadcastedParallelEnv show methods
function Base.show(io::IO, env::BroadcastedParallelEnv{E}) where {E}
    return print(io, "BroadcastedParallelEnv{", E, "}(", length(env.envs), " envs)")
end

function Base.show(io::IO, ::MIME"text/plain", env::BroadcastedParallelEnv{E}) where {E}
    println(io, "BroadcastedParallelEnv{", E, "}")
    println(io, "  - Number of environments: ", length(env.envs))
    obs_space = observation_space(env)
    act_space = action_space(env)
    println(io, "  - Observation space: ", obs_space)
    println(io, "  - Action space: ", act_space)
    print(io, "  environments: ")
    return show(io, env.envs[1])
end
