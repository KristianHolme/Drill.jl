using Test
using Drill
using Random
include("setup.jl")
using .TestSetup

@testset "GAE computation analytical verification" begin
    max_steps = 8
    gamma = 0.99f0
    gae_lambda = 0.95f0
    constant_value = 0.5f0

    env = BroadcastedParallelEnv([CustomEnv(max_steps)])

    layer = ConstantValueLayer(DrillInterface.observation_space(env), DrillInterface.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
    agent = Agent(layer, alg; verbose = 0)

    roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), gae_lambda, gamma, max_steps, 1)
    Drill.collect_rollout!(roll_buffer, agent, alg, env)

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
        layer = ConstantValueLayer(DrillInterface.observation_space(env), DrillInterface.action_space(env), constant_value)
        alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
        agent = Agent(layer, alg; verbose = 0)

        roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), gae_lambda, gamma, max_steps, 1)
        Drill.collect_rollout!(roll_buffer, agent, alg, env)

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

    layer = ConstantValueLayer(DrillInterface.observation_space(env), DrillInterface.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
    agent = Agent(layer, alg; verbose = 0)

    roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), gae_lambda, gamma, max_steps, 1)
    Drill.collect_rollout!(roll_buffer, agent, alg, env)

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

    layer = ConstantValueLayer(DrillInterface.observation_space(env), DrillInterface.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = n_total_steps, batch_size = n_total_steps, epochs = 1)
    agent = Agent(layer, alg; verbose = 0)

    roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), gae_lambda, gamma, n_total_steps, 1)
    Drill.collect_rollout!(roll_buffer, agent, alg, env)

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

    layer = ConstantValueLayer(DrillInterface.observation_space(env), DrillInterface.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
    agent = Agent(layer, alg; verbose = 0)

    roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), gae_lambda, gamma, max_steps, 1)
    Drill.collect_rollout!(roll_buffer, agent, alg, env)

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
    layer = ConstantValueLayer(DrillInterface.observation_space(env), DrillInterface.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
    agent = Agent(layer, alg; verbose = 0)

    roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), gae_lambda, gamma, max_steps, 1)
    Drill.collect_rollout!(roll_buffer, agent, alg, env)

    rewards = roll_buffer.rewards
    values = roll_buffer.values
    advantages = roll_buffer.advantages

    @test length(rewards) == 1
    @test rewards[1] ≈ 1.0f0 atol = 1.0e-5
    @test values[1] ≈ constant_value atol = 1.0e-5

    expected_advantage = 1.0f0 - constant_value
    @test advantages[1] ≈ expected_advantage atol = 1.0e-4

    gamma_zero = 0.0f0
    roll_buffer_zero = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), gae_lambda, gamma_zero, max_steps, 1)
    Drill.collect_rollout!(roll_buffer_zero, agent, alg, env)

    @test roll_buffer_zero.advantages[1] ≈ (1.0f0 - constant_value) atol = 1.0e-4

    lambda_zero = 0.0f0
    env_multi = BroadcastedParallelEnv([CustomEnv(3)])
    agent_multi = Agent(layer, alg; verbose = 0)
    roll_buffer_td0 = RolloutBuffer(DrillInterface.observation_space(env_multi), DrillInterface.action_space(env_multi), lambda_zero, gamma, 3, 1)
    Drill.collect_rollout!(roll_buffer_td0, agent_multi, alg, env_multi)

    @test all(a -> !isnan(a) && isfinite(a), roll_buffer_td0.advantages)
end
