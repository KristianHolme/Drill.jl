function is_monitored(env::AbstractParallelEnv)
    monitored = false
    while env isa AbstractParallelEnvWrapper
        monitored = env isa MonitorWrapperEnv
        env = unwrap(env)
    end
    return monitored
end

"""
    evaluate(predict_actions, rng, env; kwargs...)
    evaluate(policy::NeuralPolicy, env; kwargs...)

Roll out a policy for `n_eval_episodes` completed episodes and summarize performance.

Each finished episode contributes one **episode return** (undiscounted sum of per-step
rewards from `act!` over that episode) and one **episode length** (number of steps).
Per-step rewards from the environment are never returned directly; they are only
accumulated into these episode-level totals.

# Arguments
- `predict_actions`: Callable `(observations; deterministic, rng) -> actions` for parallel env stepping
- `rng`: Random number generator passed to `predict_actions`
- `env`: [`AbstractParallelEnv`](@ref) to evaluate on (vectorized / multi-env rollout)

# Keyword Arguments
- `n_eval_episodes::Int = 10`: Number of completed episodes to collect
- `deterministic::Bool = true`: Use deterministic actions when `true`
- `reward_threshold::Union{Nothing, Real} = nothing`: If set, error when mean episode return is below this value
- `return_stats::Bool = true`: If `true`, return aggregate statistics; if `false`, return per-episode vectors
- `warn::Bool = true`: Warn when the env is not wrapped with [`MonitorWrapperEnv`](@ref)
- `show_progress::Bool = false`: Show a progress bar while collecting episodes

# Returns
- If `return_stats = true` (default): a `NamedTuple` with
  `mean_reward`, `std_reward`, `mean_length`, `std_length`.
- If `return_stats = false`: `(episode_rewards, episode_lengths)`.
"""
function evaluate(
        predict_actions,
        rng::AbstractRNG,
        env::AbstractParallelEnv;
        n_eval_episodes::Int = 10,
        deterministic::Bool = true,
        reward_threshold::Union{Nothing, Real} = nothing,
        return_stats::Bool = true,
        warn::Bool = true,
        show_progress::Bool = false,
    )

    is_monitor_wrapped = is_monitored(env)

    if !is_monitor_wrapped && warn
        @warn """Evaluation environment is not wrapped with a Monitor wrapper. 
        This may result in reporting modified episode lengths and rewards, 
        if other wrappers happen to modify these. Consider wrapping 
        environment first with Monitor wrapper."""
    end

    T = eltype(observation_space(env))
    episode_rewards = T[]
    episode_lengths = Int[]

    n_envs = number_of_envs(env)

    current_rewards = zeros(T, n_envs)
    current_lengths = zeros(Int, n_envs)

    reset!(env)
    observations = observe(env)

    p = Progress(n_eval_episodes; enabled = show_progress)
    while length(episode_rewards) < n_eval_episodes
        actions = predict_actions(observations; deterministic, rng)

        step_rewards, terminateds, truncateds, infos = act!(env, actions)
        current_rewards .+= step_rewards
        current_lengths .+= 1
        observations = observe(env)

        dones = terminateds .| truncateds
        for i in 1:n_envs
            if length(episode_rewards) < n_eval_episodes
                if dones[i]
                    next!(p)
                    if is_monitor_wrapped && haskey(infos[i], "episode")
                        push!(episode_rewards, infos[i]["episode"]["r"])
                        push!(episode_lengths, infos[i]["episode"]["l"])
                    else
                        push!(episode_rewards, current_rewards[i])
                        push!(episode_lengths, current_lengths[i])
                    end

                    current_rewards[i] = 0
                    current_lengths[i] = 0
                end
            end
        end
    end

    mean_reward = mean(episode_rewards)
    std_reward = std(episode_rewards)

    if reward_threshold !== nothing
        if mean_reward < reward_threshold
            error("Mean reward below threshold: $(round(mean_reward, digits = 2)) < $(reward_threshold)")
        end
    end

    if return_stats
        return (; mean_reward, std_reward, mean_length = mean(episode_lengths), std_length = std(episode_lengths))
    else
        return episode_rewards, episode_lengths
    end
end

function evaluate(
        policy::NeuralPolicy,
        env::AbstractParallelEnv;
        rng::AbstractRNG = Random.default_rng(),
        kwargs...,
    )
    predict_actions(observations; deterministic, rng) = begin
        actions = policy(observations; deterministic, rng)
        return actions isa AbstractVector ? actions : collect(actions)
    end
    return evaluate(predict_actions, rng, env; kwargs...)
end

function evaluate(
        cache::RLCache,
        env::AbstractParallelEnv;
        rng::AbstractRNG = cache.rng,
        kwargs...,
    )
    predict_cache_actions(observations; deterministic, rng) = begin
        actions = predict_actions(cache, observations; deterministic, rng)
        return actions isa AbstractVector ? actions : collect(actions)
    end
    return evaluate(predict_cache_actions, rng, env; kwargs...)
end

# Temporary alias until call sites migrate from evaluate_agent.
const evaluate_agent = evaluate
