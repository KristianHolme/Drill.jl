struct MultiAgentParallelEnv{E <: AbstractParallelEnv} <: AbstractParallelEnv
    envs::Vector{E}
    env_counts::Vector{Int}  # Number of sub-envs in each parallel env
    total_envs::Int          # Sum of all env_counts

    function MultiAgentParallelEnv(envs::Vector{E}) where {E <: AbstractParallelEnv}
        @assert !isempty(envs) "Must provide at least one parallel environment"

        # All sub-environments must have the same observation and action spaces
        @assert all(env -> isequal(observation_space(env), observation_space(envs[1])), envs) "All sub-environments must have the same observation space"
        @assert all(env -> isequal(action_space(env), action_space(envs[1])), envs) "All sub-environments must have the same action space"

        env_counts = [number_of_envs(env) for env in envs]
        total_envs = sum(env_counts)

        return new{E}(envs, env_counts, total_envs)
    end
end

function reset!(env::MultiAgentParallelEnv{E}) where {E <: AbstractParallelEnv}
    reset!.(env.envs)
    return nothing
end

function observe(env::MultiAgentParallelEnv{E}) where {E <: AbstractParallelEnv}
    observations = observe.(env.envs)
    stacked_observations = vcat(observations...)
    return stacked_observations
end

function terminated(env::MultiAgentParallelEnv{E}) where {E <: AbstractParallelEnv}
    # all_terminated = Bool[]
    batch_terminated = terminated.(env.envs)
    stacked_terminated = vcat(batch_terminated...)
    return stacked_terminated
end

function truncated(env::MultiAgentParallelEnv{E}) where {E <: AbstractParallelEnv}
    batch_truncated = truncated.(env.envs)
    stacked_truncated = vcat(batch_truncated...)
    return stacked_truncated
end

function get_info(env::MultiAgentParallelEnv{E}) where {E <: AbstractParallelEnv}
    batch_infos = get_info.(env.envs)
    stacked_infos = vcat(batch_infos...)
    return stacked_infos
end

function act!(env::MultiAgentParallelEnv{E}, actions::AbstractVector) where {E <: AbstractParallelEnv}
    @assert length(actions) == number_of_envs(env) "Number of actions ($(length(actions))) must match total number of environments ($(number_of_envs(env)))"

    # Split actions into chunks for each sub-parallel-env
    idxs = cumsum([1; env.env_counts])
    chunk_indices = [idxs[i]:(idxs[i + 1] - 1) for i in 1:length(env.envs)]
    action_chunks = [actions[chunk_indices[i]] for i in 1:length(env.envs)]

    # Execute actions on each sub-parallel-env and collect results
    stacked_rewards = Vector{Float32}(undef, number_of_envs(env))
    stacked_terminated = Vector{Bool}(undef, number_of_envs(env))
    stacked_truncated = Vector{Bool}(undef, number_of_envs(env))
    stacked_infos = Vector{Dict{String, Any}}(undef, number_of_envs(env))

    @threads for i in eachindex(env.envs)
        rewards, terminateds, truncateds, infos = act!(env.envs[i], action_chunks[i])
        @assert length(rewards) == length(chunk_indices[i]) "length of rewards: $(length(rewards)) != length of chunk indices: $(length(chunk_indices[i]))"
        @assert length(terminateds) == length(chunk_indices[i]) "length of terminateds: $(length(terminateds)) != length of chunk indices: $(length(chunk_indices[i]))"
        @assert length(truncateds) == length(chunk_indices[i]) "length of truncateds: $(length(truncateds)) != length of chunk indices: $(length(chunk_indices[i]))"
        @assert length(infos) == length(chunk_indices[i]) "length of infos: $(length(infos)) != length of chunk indices: $(length(chunk_indices[i]))"
        stacked_rewards[chunk_indices[i]] .= rewards
        stacked_terminated[chunk_indices[i]] .= terminateds
        stacked_truncated[chunk_indices[i]] .= truncateds
        stacked_infos[chunk_indices[i]] .= infos
    end

    # batched_rewards = getindex.(all_returns, 1)
    # batched_terminated = getindex.(all_returns, 2)
    # batched_truncated = getindex.(all_returns, 3)
    # batched_infos = getindex.(all_returns, 4)

    # stacked_rewards = vcat(batched_rewards...)
    # stacked_terminated = vcat(batched_terminated...)
    # stacked_truncated = vcat(batched_truncated...)
    # stacked_infos = vcat(batched_infos...)

    return stacked_rewards, stacked_terminated, stacked_truncated, stacked_infos
end

function observation_space(env::MultiAgentParallelEnv{E}) where {E <: AbstractParallelEnv}
    return observation_space(env.envs[1])
end

function action_space(env::MultiAgentParallelEnv{E}) where {E <: AbstractParallelEnv}
    return action_space(env.envs[1])
end

number_of_envs(env::MultiAgentParallelEnv) = env.total_envs

function Random.seed!(env::MultiAgentParallelEnv, seed::Integer)
    for (i, sub_env) in enumerate(env.envs)
        Random.seed!(sub_env, seed + i - 1)
    end
    return env
end
unwrap_all(env::MultiAgentParallelEnv) = env.envs

# MultiAgentParallelEnv show methods
function Base.show(io::IO, env::MultiAgentParallelEnv{E}) where {E}
    return print(io, "MultiAgentParallelEnv{", E, "}(", length(env.envs), " parallel envs, ", env.total_envs, " total envs)")
end

function Base.show(io::IO, ::MIME"text/plain", env::MultiAgentParallelEnv{E}) where {E}
    println(io, "MultiAgentParallelEnv{", E, "}")
    println(io, "  - Number of parallel environments: ", length(env.envs))
    println(io, "  - Total environments: ", env.total_envs)
    println(io, "  - Environment counts per parallel env: ", env.env_counts)
    obs_space = observation_space(env)
    act_space = action_space(env)
    println(io, "  - Observation space: ", obs_space)
    println(io, "  - Action space: ", act_space)

    for (i, sub_env) in enumerate(env.envs)
        println(io, "  - Parallel env ", i, " (", env.env_counts[i], " envs): ")
        show(io, sub_env)
        if i < length(env.envs)
            println(io)
        end
    end
    return
end
