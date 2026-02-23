@testitem "GAE computation analytical verification" tags = [:gae, :analytical] setup = [SharedTestSetup] begin
    using Random

    # Test GAE computation with known values for true analytical verification
    # Manual calculation of GAE advantages for specific scenario:
    # - 8 steps, rewards = [0,0,0,0,0,0,0,1], values = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    # - γ = 0.99, λ = 0.95, episode terminates
    max_steps = 8
    gamma = 0.99f0
    gae_lambda = 0.95f0
    constant_value = 0.5f0

    # Create environment that gives reward 1.0 only at final step
    env = BroadcastedParallelEnv([SharedTestSetup.CustomEnv(max_steps)])

    # Create policy with constant value function
    policy = SharedTestSetup.ConstantValuePolicy(Drill.observation_space(env), Drill.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
    agent = Agent(policy, alg; verbose = 0)

    # Collect rollouts
    roll_buffer = RolloutBuffer(Drill.observation_space(env), Drill.action_space(env), gae_lambda, gamma, max_steps, 1)
    Drill.collect_rollout!(roll_buffer, agent, alg, env)

    # Verify reward pattern
    rewards = roll_buffer.rewards
    values = roll_buffer.values
    advantages = roll_buffer.advantages

    # Should have reward of 1.0 at final step, 0.0 elsewhere
    @test isapprox(rewards[end], 1.0f0, atol = 1.0e-5)
    @test all(r -> isapprox(r, 0.0f0, atol = 1.0e-5), rewards[1:(end - 1)])

    # Values should be constant
    @test all(v -> isapprox(v, constant_value, atol = 1.0e-5), values)

    # Manually calculated GAE advantages using the formula:
    # A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    # where δ_t = r_t + γV_{t+1} - V_t
    #
    # For this scenario:
    # δ_t = 0 + 0.99×0.5 - 0.5 = -0.005 for t=0...6
    # δ_7 = 1 + 0.99×0 - 0.5 = 0.5 for final step (terminated)
    # γλ = 0.99 × 0.95 = 0.9405

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

    # Verify returns = advantages + values
    expected_returns = expected_advantages .+ constant_value
    @test isapprox(roll_buffer.returns, expected_returns, atol = 1.0e-4)
end

@testitem "GAE computation cross-validation with different parameters" tags = [:gae, :parametric] setup = [SharedTestSetup] begin
    using Random

    # Test GAE with different gamma and lambda combinations
    max_steps = 4
    constant_value = 0.3f0

    test_cases = [
        (gamma = 0.95f0, gae_lambda = 0.9f0),
        (gamma = 0.99f0, gae_lambda = 0.95f0),
        (gamma = 1.0f0, gae_lambda = 1.0f0),   # Monte Carlo case
        (gamma = 0.9f0, gae_lambda = 0.0f0),   # TD(0) case
        (gamma = 0.8f0, gae_lambda = 0.5f0),
    ]

    for (gamma, gae_lambda) in test_cases
        env = BroadcastedParallelEnv([SharedTestSetup.CustomEnv(max_steps)])
        policy = SharedTestSetup.ConstantValuePolicy(Drill.observation_space(env), Drill.action_space(env), constant_value)
        alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
        agent = Agent(policy, alg; verbose = 0)

        roll_buffer = RolloutBuffer(Drill.observation_space(env), Drill.action_space(env), gae_lambda, gamma, max_steps, 1)
        Drill.collect_rollout!(roll_buffer, agent, alg, env)

        rewards = roll_buffer.rewards
        values = roll_buffer.values
        advantages = roll_buffer.advantages

        # Verify reward pattern
        @test rewards[end] ≈ 1.0f0 atol = 1.0e-5
        @test all(r -> isapprox(r, 0.0f0, atol = 1.0e-5), rewards[1:(end - 1)])

        # Verify constant values
        @test all(v -> isapprox(v, constant_value, atol = 1.0e-5), values)

        # Cross-validation with helper function
        expected_advantages = SharedTestSetup.compute_expected_gae(
            rewards, values, gamma, gae_lambda; is_terminated = true
        )

        @test isapprox(advantages, expected_advantages, atol = 1.0e-4)
    end
end

@testitem "GAE computation with CustomEnv" tags = [:gae, :algorithms] setup = [SharedTestSetup] begin
    using Random

    # Test parameters matching SB3 test
    max_steps = 64
    gamma = 0.99f0
    gae_lambda = 0.95f0
    constant_value = 0.5f0

    # Create environment
    env = BroadcastedParallelEnv([SharedTestSetup.CustomEnv(max_steps)])

    # Create policy with constant value function
    policy = SharedTestSetup.ConstantValuePolicy(Drill.observation_space(env), Drill.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
    agent = Agent(policy, alg; verbose = 0)

    # Collect rollouts
    roll_buffer = RolloutBuffer(Drill.observation_space(env), Drill.action_space(env), gae_lambda, gamma, max_steps, 1)
    Drill.collect_rollout!(roll_buffer, agent, alg, env)

    # Verify rewards pattern: should be [0, 0, 0, ..., 0, 1] for each episode
    rewards = roll_buffer.rewards
    values = roll_buffer.values
    advantages = roll_buffer.advantages

    # Check that rewards are mostly 0 with 1.0 at episode ends
    # Since we have constant episodes of max_steps, every max_steps-th reward should be 1.0
    for i in max_steps:max_steps:length(rewards)
        @test rewards[i] ≈ 1.0f0 atol = 1.0e-5
    end

    # Check that values are constant
    @test all(v -> isapprox(v, constant_value, atol = 1.0e-5), values)

    # Find episode boundaries and verify advantages
    episode_ends = findall(i -> i % max_steps == 0, 1:length(rewards))

    for episode_end in episode_ends
        episode_start = episode_end - max_steps + 1

        # Last step of episode should have advantage = 1.0 - constant_value
        expected_last_advantage = 1.0f0 - constant_value
        @test isapprox(advantages[episode_end], expected_last_advantage, atol = 1.0e-4)

        # Verify GAE computation for the episode
        episode_rewards = rewards[episode_start:episode_end]
        episode_values = values[episode_start:episode_end]
        episode_advantages = advantages[episode_start:episode_end]

        # Cross-validation with helper function
        expected_advantages = SharedTestSetup.compute_expected_gae(
            episode_rewards, episode_values, gamma, gae_lambda; is_terminated = true
        )

        @test isapprox(episode_advantages, expected_advantages, atol = 1.0e-4)
    end
end

@testitem "GAE computation multiple episodes" tags = [:gae, :episodes] setup = [SharedTestSetup] begin
    using Random

    # Test with multiple shorter episodes to verify episode boundary handling
    max_steps = 8
    gamma = 1.0f0  # No discounting for simpler verification
    gae_lambda = 1.0f0  # Monte Carlo returns
    constant_value = 0.0f0  # Zero baseline
    n_total_steps = 32  # Should get 4 episodes

    # Create environment
    env = BroadcastedParallelEnv([SharedTestSetup.CustomEnv(max_steps)])

    # Create policy with constant value function
    policy = SharedTestSetup.ConstantValuePolicy(Drill.observation_space(env), Drill.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = n_total_steps, batch_size = n_total_steps, epochs = 1)
    agent = Agent(policy, alg; verbose = 0)

    # Collect rollouts
    roll_buffer = RolloutBuffer(Drill.observation_space(env), Drill.action_space(env), gae_lambda, gamma, n_total_steps, 1)
    Drill.collect_rollout!(roll_buffer, agent, alg, env)

    rewards = roll_buffer.rewards
    advantages = roll_buffer.advantages
    returns = roll_buffer.returns

    # With gamma=1.0, gae_lambda=1.0, and constant_value=0.0:
    # For each episode, only the last step gets reward 1.0
    # Monte Carlo return for last step = 1.0
    # Monte Carlo return for all other steps = 1.0 (undiscounted future reward)
    # Advantages = returns - values = returns - 0.0 = returns

    # Check episode boundaries
    episode_ends = findall(isapprox.(rewards, 1.0f0, atol = 1.0e-5))

    for episode_end in episode_ends
        episode_start = max(1, episode_end - max_steps + 1)

        # For Monte Carlo with gamma=1.0, all steps in episode should have return = 1.0
        for step in episode_start:episode_end
            @test isapprox(returns[step], 1.0f0, atol = 1.0e-4)
            @test isapprox(advantages[step], 1.0f0, atol = 1.0e-4)  # Since values = 0.0
        end
    end
end

@testitem "GAE with infinite horizon environment" tags = [:gae, :infinite] setup = [SharedTestSetup] begin
    using Random

    # Test GAE computation with infinite horizon environment
    max_steps = 8
    gamma = 0.9f0
    gae_lambda = 0.8f0
    constant_value = 0.5f0

    # Create infinite horizon environment (always reward 1.0, never terminates)
    env = BroadcastedParallelEnv([SharedTestSetup.InfiniteHorizonEnv()])

    # Create policy with constant value function
    policy = SharedTestSetup.ConstantValuePolicy(Drill.observation_space(env), Drill.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
    agent = Agent(policy, alg; verbose = 0)

    # Collect rollouts
    roll_buffer = RolloutBuffer(Drill.observation_space(env), Drill.action_space(env), gae_lambda, gamma, max_steps, 1)
    Drill.collect_rollout!(roll_buffer, agent, alg, env)

    rewards = roll_buffer.rewards
    values = roll_buffer.values
    advantages = roll_buffer.advantages

    # All rewards should be 1.0 in infinite horizon env
    @test all(r -> isapprox(r, 1.0f0, atol = 1.0e-5), rewards)

    # Values should be constant
    @test all(v -> isapprox(v, constant_value, atol = 1.0e-5), values)

    # Since episode doesn't terminate, we expect bootstrapping
    # This tests the rollout-limited case where episode neither terminates nor truncates
    # but is cut off due to max_steps limit

    # For infinite horizon with constant rewards and values:
    # Each step: reward = 1.0, next_value = constant_value
    # delta = 1.0 + gamma * constant_value - constant_value = 1.0 + (gamma - 1) * constant_value
    # The GAE computation should handle the bootstrap value from the last observation

    # Verify the advantage computation makes sense
    # Last step should bootstrap with next state value
    # For infinite horizon, this should be approximately the same as constant_value
    expected_delta = 1.0f0 + (gamma - 1.0f0) * constant_value

    # The actual computation depends on bootstrapping, but we can verify basic properties
    @test !isapprox(advantages[1], advantages[end], atol = 1.0e-6)  # Should vary due to GAE
    @test all(a -> !isnan(a) && isfinite(a), advantages)  # All advantages should be finite
end

@testitem "GAE edge cases" tags = [:gae, :edge_cases] setup = [SharedTestSetup] begin
    using Random

    # Test edge cases for GAE computation

    # Single step episode
    max_steps = 1
    gamma = 0.9f0
    gae_lambda = 0.8f0
    constant_value = 0.3f0

    env = BroadcastedParallelEnv([SharedTestSetup.CustomEnv(max_steps)])
    policy = SharedTestSetup.ConstantValuePolicy(Drill.observation_space(env), Drill.action_space(env), constant_value)
    alg = PPO(; gamma, gae_lambda, n_steps = max_steps, batch_size = max_steps, epochs = 1)
    agent = Agent(policy, alg; verbose = 0)

    roll_buffer = RolloutBuffer(Drill.observation_space(env), Drill.action_space(env), gae_lambda, gamma, max_steps, 1)
    Drill.collect_rollout!(roll_buffer, agent, alg, env)

    rewards = roll_buffer.rewards
    values = roll_buffer.values
    advantages = roll_buffer.advantages

    # Single step: reward = 1.0, value = constant_value
    @test length(rewards) == 1
    @test rewards[1] ≈ 1.0f0 atol = 1.0e-5
    @test values[1] ≈ constant_value atol = 1.0e-5

    # Advantage should be reward - value = 1.0 - constant_value
    expected_advantage = 1.0f0 - constant_value
    @test advantages[1] ≈ expected_advantage atol = 1.0e-4

    # Test zero gamma case
    gamma_zero = 0.0f0
    roll_buffer_zero = RolloutBuffer(Drill.observation_space(env), Drill.action_space(env), gae_lambda, gamma_zero, max_steps, 1)
    Drill.collect_rollout!(roll_buffer_zero, agent, alg, env)

    # With gamma=0, advantage should just be immediate reward - value
    @test roll_buffer_zero.advantages[1] ≈ (1.0f0 - constant_value) atol = 1.0e-4

    # Test zero lambda case (TD(0))
    lambda_zero = 0.0f0
    env_multi = BroadcastedParallelEnv([SharedTestSetup.CustomEnv(3)])  # 3 steps for better testing
    agent_multi = Agent(policy, alg; verbose = 0)
    roll_buffer_td0 = RolloutBuffer(Drill.observation_space(env_multi), Drill.action_space(env_multi), lambda_zero, gamma, 3, 1)
    Drill.collect_rollout!(roll_buffer_td0, agent_multi, alg, env_multi)

    # With lambda=0 (TD(0)), GAE reduces to simple TD error
    # Each advantage should be just the immediate TD error without accumulation
    @test all(a -> !isnan(a) && isfinite(a), roll_buffer_td0.advantages)
end
