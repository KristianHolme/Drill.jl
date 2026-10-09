# The parallel env contract implemented over a vector of single envs, shared by
# BroadcastedParallelEnv and MultiThreadedParallelEnv. Each type defines
# `_foreach_env(f, penv)`, which calls `f(i)` for every env index.

const VectorParallelEnv = Union{BroadcastedParallelEnv, MultiThreadedParallelEnv}

function _check_homogeneous(envs::Vector{E}) where {E <: AbstractEnv}
    @assert !isempty(envs) "Must provide at least one environment"
    @assert all(env -> typeof(env) == E, envs) "All environments must be of the same type"
    @assert all(env -> isequal(observation_space(env), observation_space(envs[1])), envs) "All environments must have the same observation space"
    @assert all(env -> isequal(action_space(env), action_space(envs[1])), envs) "All environments must have the same action space"
    return nothing
end

number_of_envs(penv::VectorParallelEnv) = length(penv.envs)
observation_space(penv::VectorParallelEnv) = observation_space(penv.envs[1])
action_space(penv::VectorParallelEnv) = action_space(penv.envs[1])

"""
    reward_type(space)

The element type used for reward vectors: `eltype(space)` for floating-point spaces,
`Float32` otherwise.
"""
reward_type(space::AbstractSpace) = eltype(space) <: AbstractFloat ? eltype(space) : Float32

_collect_infos(infos) = all(isnothing, infos) ? nothing : infos

_sub_seed(::Nothing, offset::Int) = nothing
_sub_seed(seed::Integer, offset::Int) = seed + offset

function observe(penv::VectorParallelEnv)
    obs = allocate_observations(observation_space(penv), number_of_envs(penv))
    _foreach_env(penv) do i
        observe!(observation_slot(obs, i), penv.envs[i])
    end
    return obs
end

function reset!(penv::VectorParallelEnv; seed::Union{Nothing, Integer} = nothing)
    _foreach_env(penv) do i
        reset!(penv.envs[i]; seed = _sub_seed(seed, i - 1))
    end
    return observe(penv)
end

function step!(penv::VectorParallelEnv, actions::AbstractVector)
    n = number_of_envs(penv)
    @assert length(actions) == n "Number of actions ($(length(actions))) must match number of environments ($n)"
    space = observation_space(penv)
    rewards = Vector{reward_type(space)}(undef, n)
    terminateds = Vector{Bool}(undef, n)
    truncateds = Vector{Bool}(undef, n)
    infos = Vector{Any}(undef, n)
    obs = allocate_observations(space, n)
    final_obs = allocate_observations(space, n)
    _foreach_env(penv) do i
        env = penv.envs[i]
        rewards[i] = act!(env, actions[i])
        terminateds[i] = terminated(env)
        truncateds[i] = truncated(env)
        infos[i] = get_info(env)
        observe!(observation_slot(final_obs, i), env)
        if terminateds[i] || truncateds[i]
            reset!(env)
            observe!(observation_slot(obs, i), env)
        else
            observation_slot(obs, i) .= observation_slot(final_obs, i)
        end
    end
    return obs, rewards, terminateds, truncateds, final_obs, _collect_infos(infos)
end
