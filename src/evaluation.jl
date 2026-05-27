function is_monitored(env::AbstractParallelEnv)
    monitored = false
    while env isa AbstractParallelEnvWrapper
        monitored = env isa MonitorWrapperEnv
        env = unwrap(env)
    end
    return monitored
end

"""
    evaluate_agent(agent, env; kwargs...)

Roll out a policy for `n_eval_episodes` completed episodes and summarize performance.

Each finished episode contributes one **episode return** (undiscounted sum of per-step
rewards from `act!` over that episode) and one **episode length** (number of steps).
Per-step rewards from the environment are never returned directly; they are only
accumulated into these episode-level totals.

# Arguments
- `agent`: Agent whose policy is evaluated (must support `predict_actions`)
- `env`: [`AbstractParallelEnv`](@ref) to evaluate on (vectorized / multi-env rollout)

# Keyword Arguments
- `n_eval_episodes::Int = 10`: Number of completed episodes to collect
- `deterministic::Bool = true`: Use deterministic actions when `true`
- `reward_threshold::Union{Nothing, Real} = nothing`: If set, error when mean episode return is below this value
- `return_stats::Bool = true`: If `true`, return aggregate statistics; if `false`, return per-episode vectors
- `warn::Bool = true`: Warn when the env is not wrapped with [`MonitorWrapperEnv`](@ref)
- `rng::AbstractRNG = agent.rng`: RNG passed to `predict_actions`
- `show_progress::Bool = false`: Show a progress bar while collecting episodes

# Episode returns and lengths
For every completed episode, evaluation records:
- **Return** `r`: ∑ₜ rewardₜ, the sum of scalar rewards returned by `act!` for each step in that episode (no discounting).
- **Length** `l`: number of steps in that episode.

How those totals are obtained:
- **With [`MonitorWrapperEnv`](@ref)**: on episode end, `r` and `l` are read from `infos[i]["episode"]`
  (the same undiscounted return and step count the monitor accumulated from per-step rewards).
- **Without a monitor**: the same quantities are computed inside `evaluate_agent` by adding each
  step's reward to `current_rewards` and incrementing `current_lengths` until the episode terminates.

Aggregate outputs (`mean_reward`, `std_reward`, etc.) are means and standard deviations **across
episodes** of those episode returns and lengths, not across individual steps.

# Returns
- If `return_stats = true` (default): a `NamedTuple` with
  `mean_reward`, `std_reward`, `mean_length`, `std_length`.
  Reward statistics use `eltype(observation_space(env))`; lengths are `Int`-based episode step counts.
- If `return_stats = false`: `(episode_rewards, episode_lengths)` where
  `episode_rewards::Vector{T}` has one **episode return** per completed episode (length `n_eval_episodes`)
  and `episode_lengths::Vector{Int}` has the matching step counts.

# Notes
- Episodes are collected across parallel sub-environments until `n_eval_episodes` completions are recorded.
- Wrappers that alter per-step rewards or episode boundaries can change reported returns; prefer
  [`MonitorWrapperEnv`](@ref) when wrappers sit between the base env and evaluation.

# Examples
```julia
# Aggregate statistics (default)
stats = evaluate_agent(agent, env; n_eval_episodes = 20)
stats.mean_reward, stats.std_reward

# Or destructure the four summary values
mean_reward, std_reward, mean_length, std_length =
    evaluate_agent(agent, env; n_eval_episodes = 20)

# Per-episode vectors (each episode_rewards[i] is that episode's undiscounted return sum)
episode_rewards, episode_lengths = evaluate_agent(
    agent, env; n_eval_episodes = 20, return_stats = false
)

# Fail if mean episode return is too low
evaluate_agent(agent, env; reward_threshold = 100.0, n_eval_episodes = 50)
```
"""
function evaluate_agent(
        agent,
        env::AbstractParallelEnv;
        n_eval_episodes::Int = 10,
        deterministic::Bool = true,
        reward_threshold::Union{Nothing, Real} = nothing,
        return_stats::Bool = true,
        warn::Bool = true,
        rng::AbstractRNG = agent.rng,
        show_progress::Bool = false
    )

    # Check if environment is wrapped with Monitor (when Monitor is implemented)
    is_monitor_wrapped = is_monitored(env)

    if !is_monitor_wrapped && warn
        @warn """Evaluation environment is not wrapped with a Monitor wrapper. 
        This may result in reporting modified episode lengths and rewards, 
        if other wrappers happen to modify these. Consider wrapping 
        environment first with Monitor wrapper."""
    end

    # Initialize tracking variables
    T = eltype(observation_space(env))
    episode_rewards = T[]
    episode_lengths = Int[]

    n_envs = number_of_envs(env)
    # For parallel environments, distribute episodes evenly

    current_rewards = zeros(T, n_envs)
    current_lengths = zeros(Int, n_envs)

    # Reset environment
    reset!(env)
    observations = observe(env)

    p = Progress(n_eval_episodes; enabled = show_progress)
    while length(episode_rewards) < n_eval_episodes
        # Get actions from agent
        actions = predict_actions(agent, observations; deterministic, rng)

        # Take step in environment
        step_rewards, terminateds, truncateds, infos = act!(env, actions)
        current_rewards .+= step_rewards
        current_lengths .+= 1
        observations = observe(env)

        # Process each environment
        dones = terminateds .| truncateds
        for i in 1:n_envs
            if length(episode_rewards) < n_eval_episodes
                # Check if episode ended
                if dones[i]
                    next!(p)
                    if is_monitor_wrapped && haskey(infos[i], "episode")
                        # Use Monitor statistics if available
                        push!(episode_rewards, infos[i]["episode"]["r"])
                        push!(episode_lengths, infos[i]["episode"]["l"])
                    else
                        # Use manually tracked statistics
                        push!(episode_rewards, current_rewards[i])
                        push!(episode_lengths, current_lengths[i])
                    end

                    current_rewards[i] = 0
                    current_lengths[i] = 0
                end
            end
        end
    end

    # Calculate statistics
    mean_reward = mean(episode_rewards)
    std_reward = std(episode_rewards)

    # Check reward threshold if specified
    if reward_threshold !== nothing
        if mean_reward < reward_threshold
            error("Mean reward below threshold: $(round(mean_reward, digits = 2)) < $(reward_threshold)")
        end
    end

    # Return results
    if return_stats
        return (; mean_reward, std_reward, mean_length = mean(episode_lengths), std_length = std(episode_lengths))
    else
        return episode_rewards, episode_lengths
    end
end
