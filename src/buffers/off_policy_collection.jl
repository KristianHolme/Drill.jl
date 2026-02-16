# Generic off-policy collection functions

# Helper function to get raw actions from agent (for off-policy algorithms)
# Default: agents don't support raw actions
# Agent-specific implementations (e.g., for OffPolicyActorCriticAgent) should be defined in their respective algorithm files
function predict_actions_raw(agent::AbstractAgent, observations::AbstractVector)
    error("Agent $(typeof(agent)) does not support raw actions. Use an off-policy actor-critic agent.")
end

"""
    collect_trajectories(agent, alg, env, n_steps, [progress_meter]; kwargs...)

Collect off-policy trajectories for any agent and off-policy algorithm.

# Arguments
- `agent::AbstractAgent`: The agent (must support `predict_actions` with optional `raw` kwarg)
- `alg::OffPolicyAlgorithm`: The off-policy algorithm
- `env::AbstractParallelEnv`: The parallel environment
- `n_steps::Int`: Number of steps to collect
- `progress_meter::Union{Progress, Nothing}`: Optional progress meter
- `callbacks::Union{Vector{<:AbstractCallback}, Nothing}`: Optional callbacks
- `use_random_actions::Bool`: Whether to use random actions instead of agent predictions

# Returns
- `trajectories::Vector{OffPolicyTrajectory}`: Collected trajectories
- `success::Bool`: Whether collection completed successfully
"""
function collect_trajectories(
        agent::AbstractAgent,
        alg::OffPolicyAlgorithm,
        env::AbstractParallelEnv,
        n_steps::Int,
        progress_meter::Union{Progress, Nothing} = nothing;
        callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing,
        use_random_actions::Bool = false
    )

    trajectories = OffPolicyTrajectory[]
    obs_space = observation_space(env)
    act_space = action_space(env)
    n_envs = number_of_envs(env)
    current_trajectories = [OffPolicyTrajectory(obs_space, act_space) for _ in 1:n_envs]
    new_obs = observe(env)
    for i in 1:n_steps
        if !isnothing(callbacks)
            if !all(c -> on_step(c, Base.@locals), callbacks)
                @warn "Collecting trajectories stopped due to callback failure"
                return trajectories, false
            end
        end
        observations = new_obs
        if use_random_actions
            @assert observations isa AbstractVector
            actions = rand(agent.rng, act_space, length(observations))
            processed_actions = actions  # already in env space
        else
            # Get raw actions - will dispatch to agent-specific methods that support raw parameter
            actions_batched = predict_actions_raw(agent, observations)
            actions = actions_batched isa AbstractVector ? collect(actions_batched) : collect(eachslice(actions_batched, dims = ndims(actions_batched)))
            adapter = agent.action_adapter
            processed_actions = to_env.(Ref(adapter), actions, Ref(act_space))
        end
        rewards, terminateds, truncateds, infos = act!(env, processed_actions)
        new_obs = observe(env)
        for j in 1:n_envs
            push!(current_trajectories[j].observations, observations[j])
            push!(current_trajectories[j].actions, actions[j]) #store unprocessed actions
            push!(current_trajectories[j].rewards, rewards[j])
            if terminateds[j] || truncateds[j] || i == n_steps
                current_trajectories[j].terminated = terminateds[j]
                current_trajectories[j].truncated = truncateds[j]

                # Handle bootstrapping for truncated episodes
                if truncateds[j] && haskey(infos[j], "terminal_observation")
                    last_observation = infos[j]["terminal_observation"]
                    current_trajectories[j].truncated_observation = last_observation
                end

                # Handle bootstrapping for rollout-limited trajectories (neither terminated nor truncated)
                # We need to bootstrap with the value of the current observation
                if !terminateds[j] && !truncateds[j] && i == n_steps
                    # Get the next observation after last step (which is the current state)
                    next_obs = new_obs[j]
                    current_trajectories[j].truncated_observation = next_obs
                end

                push!(trajectories, current_trajectories[j])
                current_trajectories[j] = OffPolicyTrajectory(obs_space, act_space)
            end
        end
        !isnothing(progress_meter) && next!(progress_meter, step = number_of_envs(env))
    end
    return trajectories, true
end

"""
    collect_rollout!(buffer, agent, alg, env, n_steps, [progress_meter]; kwargs...)

Collect off-policy rollout and add to replay buffer.

# Arguments
- `buffer::ReplayBuffer`: The replay buffer to fill
- `agent::AbstractAgent`: The agent
- `alg::OffPolicyAlgorithm`: The off-policy algorithm
- `env::AbstractParallelEnv`: The parallel environment
- `n_steps::Int`: Number of steps to collect
- `progress_meter::Union{Progress, Nothing}`: Optional progress meter
- `kwargs...`: Additional arguments passed to `collect_trajectories`

# Returns
- `fps::Float64`: Frames per second during collection
- `success::Bool`: Whether collection completed successfully
"""
function collect_rollout!(
        buffer::ReplayBuffer,
        agent::AbstractAgent,
        alg::OffPolicyAlgorithm,
        env::AbstractParallelEnv,
        n_steps::Int,
        progress_meter::Union{Progress, Nothing} = nothing;
        kwargs...
    )
    t_start = time()
    trajectories, success = collect_trajectories(agent, alg, env, n_steps, progress_meter; kwargs...)
    t_collect = time() - t_start
    total_steps = sum(length.(trajectories))
    fps = total_steps / t_collect
    if !success
        @warn "Collecting trajectories stopped due to callback failure"
        return fps, false
    end

    for traj in trajectories
        push!(buffer, traj)
    end
    return fps, true
end
