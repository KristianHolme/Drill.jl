using Test
using Drill
using DrillInterface
using Random
include("setup.jl")
using .TestSetup

function make_cache(env, layer, alg; max_steps = alg.n_steps * DrillInterface.number_of_envs(env))
    prob = RLProblem(env, layer)
    return init(prob, alg; max_steps, verbosity = 0)
end

function collect_and_prepare!(roll_buffer, cache, alg, env)
    Drill.collect_rollout!(roll_buffer, cache, alg, env)
    Drill.prepare_rollout!(roll_buffer, alg)
    return roll_buffer
end

@testset "GAE computation analytical verification" begin
    max_steps = 8
    gamma = 0.99f0
    gae_lambda = 0.95f0
    constant_value = 0.5f0

    env = BroadcastedParallelEnv([CustomEnv(max_steps)])

    layer = ConstantValueModel(DrillInterface.observation_space(env), DrillInterface.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
    cache = make_cache(env, layer, alg)

    roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), max_steps, 1)
    collect_and_prepare!(roll_buffer, cache, alg, env)

    rewards = roll_buffer.rewards
    values = roll_buffer.values
    advantages = roll_buffer.advantages

    @test isapprox(rewards[end], 1.0f0, atol = 1.0e-5)
    @test all(r -> isapprox(r, 0.0f0, atol = 1.0e-5), rewards[1:(end - 1)])

    @test all(v -> isapprox(v, constant_value, atol = 1.0e-5), values)

    δ_7 = 1 + 0.99 * 0 - 0.5
    δ_t = -0.005
    γλ = 0.99 * 0.95
    A_s1 = zeros(Float32, 8)

    A_s1[end] = δ_7
    for i in 7:-1:1
        A_s1[i] = δ_t + γλ * A_s1[i + 1]
    end

    A_s2 = zeros(Float32, 8)
    A_s2[end] = δ_7
    for ix in 1:7
        i = ix - 1
        A_s2[ix] = δ_t * ((1 - γλ^(6 - i + 1)) / (1 - γλ)) + γλ^(6 - i + 1) * 0.5f0
    end
    @test isapprox(A_s1, A_s2, atol = 1.0e-4)

    expected_advantages = (A_s1 .+ A_s2) ./ 2.0f0

    @test isapprox(advantages, expected_advantages, atol = 1.0e-4)

    expected_returns = expected_advantages .+ constant_value
    @test isapprox(roll_buffer.returns, expected_returns, atol = 1.0e-4)
end

@testset "GAE computation cross-validation with different parameters" begin
    max_steps = 4
    constant_value = 0.3f0

    test_cases = [
        (gamma = 0.95f0, gae_lambda = 0.9f0),
        (gamma = 0.99f0, gae_lambda = 0.95f0),
        (gamma = 1.0f0, gae_lambda = 1.0f0),
        (gamma = 0.9f0, gae_lambda = 0.0f0),
        (gamma = 0.8f0, gae_lambda = 0.5f0),
    ]

    for (gamma, gae_lambda) in test_cases
        env = BroadcastedParallelEnv([CustomEnv(max_steps)])
        layer = ConstantValueModel(DrillInterface.observation_space(env), DrillInterface.action_space(env), constant_value)
        alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
        cache = make_cache(env, layer, alg)

        roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), max_steps, 1)
        collect_and_prepare!(roll_buffer, cache, alg, env)

        rewards = roll_buffer.rewards
        values = roll_buffer.values
        advantages = roll_buffer.advantages

        @test rewards[end] ≈ 1.0f0 atol = 1.0e-5
        @test all(r -> isapprox(r, 0.0f0, atol = 1.0e-5), rewards[1:(end - 1)])

        @test all(v -> isapprox(v, constant_value, atol = 1.0e-5), values)

        expected_advantages = compute_expected_gae(
            rewards, values, gamma, gae_lambda; is_terminated = true
        )

        @test isapprox(advantages, expected_advantages, atol = 1.0e-4)
    end
end

@testset "GAE computation with CustomEnv" begin
    max_steps = 64
    gamma = 0.99f0
    gae_lambda = 0.95f0
    constant_value = 0.5f0

    env = BroadcastedParallelEnv([CustomEnv(max_steps)])

    layer = ConstantValueModel(DrillInterface.observation_space(env), DrillInterface.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
    cache = make_cache(env, layer, alg)

    roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), max_steps, 1)
    collect_and_prepare!(roll_buffer, cache, alg, env)

    rewards = roll_buffer.rewards
    values = roll_buffer.values
    advantages = roll_buffer.advantages

    for i in max_steps:max_steps:length(rewards)
        @test rewards[i] ≈ 1.0f0 atol = 1.0e-5
    end

    @test all(v -> isapprox(v, constant_value, atol = 1.0e-5), values)

    episode_ends = findall(i -> i % max_steps == 0, 1:length(rewards))

    for episode_end in episode_ends
        episode_start = episode_end - max_steps + 1

        expected_last_advantage = 1.0f0 - constant_value
        @test isapprox(advantages[episode_end], expected_last_advantage, atol = 1.0e-4)

        episode_rewards = rewards[episode_start:episode_end]
        episode_values = values[episode_start:episode_end]
        episode_advantages = advantages[episode_start:episode_end]

        expected_advantages = compute_expected_gae(
            episode_rewards, episode_values, gamma, gae_lambda; is_terminated = true
        )

        @test isapprox(episode_advantages, expected_advantages, atol = 1.0e-4)
    end
end

@testset "GAE computation multiple episodes" begin
    max_steps = 8
    gamma = 1.0f0
    gae_lambda = 1.0f0
    constant_value = 0.0f0
    n_total_steps = 32

    env = BroadcastedParallelEnv([CustomEnv(max_steps)])

    layer = ConstantValueModel(DrillInterface.observation_space(env), DrillInterface.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = n_total_steps, batch_size = n_total_steps, epochs = 1)
    cache = make_cache(env, layer, alg)

    roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), n_total_steps, 1)
    collect_and_prepare!(roll_buffer, cache, alg, env)

    rewards = roll_buffer.rewards
    advantages = roll_buffer.advantages
    returns = roll_buffer.returns

    episode_ends = findall(isapprox.(rewards, 1.0f0, atol = 1.0e-5))

    for episode_end in episode_ends
        episode_start = max(1, episode_end - max_steps + 1)

        for step in episode_start:episode_end
            @test isapprox(returns[step], 1.0f0, atol = 1.0e-4)
            @test isapprox(advantages[step], 1.0f0, atol = 1.0e-4)
        end
    end
end

@testset "GAE with infinite horizon environment" begin
    max_steps = 8
    gamma = 0.9f0
    gae_lambda = 0.8f0
    constant_value = 0.5f0

    env = BroadcastedParallelEnv([InfiniteHorizonEnv()])

    layer = ConstantValueModel(DrillInterface.observation_space(env), DrillInterface.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
    cache = make_cache(env, layer, alg)

    roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), max_steps, 1)
    collect_and_prepare!(roll_buffer, cache, alg, env)

    rewards = roll_buffer.rewards
    values = roll_buffer.values
    advantages = roll_buffer.advantages

    @test all(r -> isapprox(r, 1.0f0, atol = 1.0e-5), rewards)

    @test all(v -> isapprox(v, constant_value, atol = 1.0e-5), values)

    expected_delta = 1.0f0 + (gamma - 1.0f0) * constant_value

    @test !isapprox(advantages[1], advantages[end], atol = 1.0e-6)
    @test all(a -> !isnan(a) && isfinite(a), advantages)
end

@testset "GAE edge cases" begin
    max_steps = 1
    gamma = 0.9f0
    gae_lambda = 0.8f0
    constant_value = 0.3f0

    env = BroadcastedParallelEnv([CustomEnv(max_steps)])
    layer = ConstantValueModel(DrillInterface.observation_space(env), DrillInterface.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
    cache = make_cache(env, layer, alg)

    roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), max_steps, 1)
    collect_and_prepare!(roll_buffer, cache, alg, env)

    rewards = roll_buffer.rewards
    values = roll_buffer.values
    advantages = roll_buffer.advantages

    @test length(rewards) == 1
    @test rewards[1] ≈ 1.0f0 atol = 1.0e-5
    @test values[1] ≈ constant_value atol = 1.0e-5

    expected_advantage = 1.0f0 - constant_value
    @test advantages[1] ≈ expected_advantage atol = 1.0e-4

    gamma_zero = 0.0f0
    roll_buffer_zero = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), max_steps, 1)
    Drill.collect_rollout!(roll_buffer_zero, cache, alg, env)
    Drill.compute_gae!(roll_buffer_zero, gamma_zero, gae_lambda)

    @test roll_buffer_zero.advantages[1] ≈ (1.0f0 - constant_value) atol = 1.0e-4

    lambda_zero = 0.0f0
    env_multi = BroadcastedParallelEnv([CustomEnv(3)])
    cache_multi = make_cache(env_multi, layer, alg)
    roll_buffer_td0 = RolloutBuffer(DrillInterface.observation_space(env_multi), DrillInterface.action_space(env_multi), 3, 1)
    Drill.collect_rollout!(roll_buffer_td0, cache_multi, alg, env_multi)
    Drill.compute_gae!(roll_buffer_td0, gamma, lambda_zero)

    @test all(a -> !isnan(a) && isfinite(a), roll_buffer_td0.advantages)
end

# Buffer for `compute_gae!` tests with per-env inputs given as `n_envs × n_steps` matrices.
function gae_buffer(rewards, values, terminateds, truncateds, bootstrap_values)
    n_envs, n_steps = size(rewards)
    obs_space = Box(Float32[-1.0], Float32[1.0])
    act_space = Box(Float32[-1.0], Float32[1.0])
    buffer = RolloutBuffer(obs_space, act_space, n_steps, n_envs)
    for t in 1:n_steps
        store_step!(
            buffer, t, zeros(Float32, 1, n_envs), zeros(Float32, 1, n_envs),
            rewards[:, t], zeros(Float32, n_envs), values[:, t],
            terminateds[:, t], truncateds[:, t],
        )
        buffer.bootstrap_values[step_indices(buffer, t)] .= bootstrap_values[:, t]
    end
    return buffer
end

# Reference advantages for one env: split its steps into episode segments and apply
# `compute_expected_gae` to each segment.
function reference_env_gae(rewards, values, terminateds, truncateds, bootstrap_values, gamma, gae_lambda)
    n_steps = length(rewards)
    advantages = zeros(Float32, n_steps)
    start = 1
    for t in 1:n_steps
        if terminateds[t] || truncateds[t] || t == n_steps
            seg = start:t
            advantages[seg] .= compute_expected_gae(
                rewards[seg], values[seg], gamma, gae_lambda;
                is_terminated = terminateds[t], bootstrap_value = bootstrap_values[t],
            )
            start = t + 1
        end
    end
    return advantages
end

function check_gae(rewards, values, terminateds, truncateds, bootstrap_values, gamma, gae_lambda)
    buffer = gae_buffer(rewards, values, terminateds, truncateds, bootstrap_values)
    compute_gae!(buffer, gamma, gae_lambda)
    n_envs, n_steps = size(rewards)
    for e in 1:n_envs
        inds = [step_indices(buffer, t)[e] for t in 1:n_steps]
        expected = reference_env_gae(
            rewards[e, :], values[e, :], terminateds[e, :], truncateds[e, :],
            bootstrap_values[e, :], gamma, gae_lambda,
        )
        @test buffer.advantages[inds] ≈ expected atol = 1.0e-5
        @test buffer.returns[inds] ≈ expected .+ values[e, :] atol = 1.0e-5
    end
    return buffer
end

@testset "compute_gae! single env with several terminated episodes" begin
    rng = Random.Xoshiro(1)
    n_steps = 10
    rewards = rand(rng, Float32, 1, n_steps)
    values = rand(rng, Float32, 1, n_steps)
    terminateds = falses(1, n_steps)
    terminateds[1, [3, 7]] .= true
    truncateds = falses(1, n_steps)
    bootstrap_values = zeros(Float32, 1, n_steps)
    bootstrap_values[1, n_steps] = 0.4f0
    buffer = check_gae(rewards, values, terminateds, truncateds, bootstrap_values, 0.9f0, 0.8f0)

    # The step before a termination does not look past the episode boundary.
    @test buffer.advantages[3] ≈ rewards[3] - values[3] atol = 1.0e-6
    @test buffer.advantages[7] ≈ rewards[7] - values[7] atol = 1.0e-6
    # The rollout ends mid-episode, so the last step bootstraps.
    @test buffer.advantages[n_steps] ≈ rewards[n_steps] + 0.9f0 * 0.4f0 - values[n_steps] atol = 1.0e-6
end

@testset "compute_gae! multiple envs with different episode boundaries" begin
    rng = Random.Xoshiro(2)
    n_envs, n_steps = 3, 8
    rewards = rand(rng, Float32, n_envs, n_steps)
    values = rand(rng, Float32, n_envs, n_steps)
    terminateds = falses(n_envs, n_steps)
    truncateds = falses(n_envs, n_steps)
    terminateds[1, 2] = true
    terminateds[1, 6] = true
    truncateds[2, 4] = true
    terminateds[3, n_steps] = true
    bootstrap_values = zeros(Float32, n_envs, n_steps)
    bootstrap_values[2, 4] = 0.7f0
    bootstrap_values[1, n_steps] = 0.3f0
    bootstrap_values[2, n_steps] = -0.2f0
    for (gamma, gae_lambda) in ((0.99f0, 0.95f0), (0.9f0, 0.0f0), (1.0f0, 1.0f0), (0.0f0, 0.8f0))
        check_gae(rewards, values, terminateds, truncateds, bootstrap_values, gamma, gae_lambda)
    end
end

@testset "compute_gae! truncation uses bootstrap value" begin
    gamma, gae_lambda = 0.9f0, 0.8f0
    rewards = Float32[1 1 1 1]
    values = Float32[0.5 0.5 0.5 0.5]
    terminateds = falses(1, 4)
    truncateds = falses(1, 4)
    truncateds[1, 2] = true
    bootstrap_values = Float32[0 2 0 0]
    buffer = check_gae(rewards, values, terminateds, truncateds, bootstrap_values, gamma, gae_lambda)
    @test buffer.advantages[2] ≈ 1.0f0 + gamma * 2.0f0 - 0.5f0 atol = 1.0e-6

    # The same step marked terminated ignores the bootstrap value.
    buffer_term = gae_buffer(rewards, values, truncateds, terminateds, bootstrap_values)
    compute_gae!(buffer_term, gamma, gae_lambda)
    @test buffer_term.advantages[2] ≈ 0.5f0 atol = 1.0e-6
    @test !(buffer_term.advantages[1] ≈ buffer.advantages[1])
end

@testset "compute_gae! rollout end with and without termination" begin
    gamma, gae_lambda = 0.95f0, 0.9f0
    rewards = Float32[0 0 1; 0 0 1]
    values = Float32[0.2 0.3 0.4; 0.2 0.3 0.4]
    terminateds = falses(2, 3)
    terminateds[2, 3] = true
    truncateds = falses(2, 3)
    # Env 2 terminates at the last step; a stray bootstrap value must be ignored.
    bootstrap_values = Float32[0 0 1.5; 0 0 1.5]
    buffer = check_gae(rewards, values, terminateds, truncateds, bootstrap_values, gamma, gae_lambda)
    last_inds = step_indices(buffer, 3)
    @test buffer.advantages[last_inds[1]] ≈ 1.0f0 + gamma * 1.5f0 - 0.4f0 atol = 1.0e-6
    @test buffer.advantages[last_inds[2]] ≈ 1.0f0 - 0.4f0 atol = 1.0e-6
end

@testset "compute_gae! lambda = 0 is TD(0)" begin
    rng = Random.Xoshiro(3)
    n_envs, n_steps = 2, 5
    gamma = 0.9f0
    rewards = rand(rng, Float32, n_envs, n_steps)
    values = rand(rng, Float32, n_envs, n_steps)
    terminateds = falses(n_envs, n_steps)
    terminateds[1, 3] = true
    truncateds = falses(n_envs, n_steps)
    bootstrap_values = zeros(Float32, n_envs, n_steps)
    bootstrap_values[:, n_steps] .= 0.6f0
    buffer = check_gae(rewards, values, terminateds, truncateds, bootstrap_values, gamma, 0.0f0)
    for e in 1:n_envs, t in 1:n_steps
        i = step_indices(buffer, t)[e]
        next_value = if terminateds[e, t]
            0.0f0
        elseif t == n_steps
            bootstrap_values[e, t]
        else
            values[e, t + 1]
        end
        @test buffer.advantages[i] ≈ rewards[e, t] + gamma * next_value - values[e, t] atol = 1.0e-6
    end
end
