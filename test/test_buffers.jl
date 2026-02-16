using TestItems

@testitem "Buffer logprobs consistency" tags = [:buffers, :rollouts] setup = [SharedTestSetup] begin
    using ClassicControlEnvironments
    using Random

    # Test: logprobs are consistent after rollout
    pend_env() = PendulumEnv()
    env = MultiThreadedParallelEnv([pend_env() for _ in 1:4])
    policy = ActorCriticLayer(DRiL.observation_space(env), DRiL.action_space(env))
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 1)
    agent = Agent(policy, alg; verbose = 0)
    n_steps = alg.n_steps
    n_envs = DRiL.number_of_envs(env)
    roll_buffer = RolloutBuffer(DRiL.observation_space(env), DRiL.action_space(env), alg.gae_lambda, alg.gamma, n_steps, n_envs)

    for i in 1:10
        DRiL.collect_rollout!(roll_buffer, agent, alg, env)
        obs = roll_buffer.observations
        act = roll_buffer.actions
        logprobs = roll_buffer.logprobs
        ps = agent.train_state.parameters
        st = agent.train_state.states
        _, new_logprobs, _, _ = DRiL.evaluate_actions(policy, obs, act, ps, st)
        @test isapprox(vec(logprobs), vec(new_logprobs); atol = 1.0e-5, rtol = 1.0e-5)
    end
end

@testitem "Buffer reset functionality" tags = [:buffers] setup = [SharedTestSetup] begin
    using Random

    # Test buffer reset clears all data
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    act_space = Box(Float32[-1.0], Float32[1.0])

    roll_buffer = RolloutBuffer(obs_space, act_space, 0.95f0, 0.99f0, 8, 2)

    # Fill buffer with some data
    roll_buffer.observations .= 1.0f0
    roll_buffer.actions .= 2.0f0
    roll_buffer.rewards .= 3.0f0
    roll_buffer.advantages .= 4.0f0
    roll_buffer.returns .= 5.0f0
    roll_buffer.logprobs .= 6.0f0
    roll_buffer.values .= 7.0f0

    # Reset buffer
    DRiL.reset!(roll_buffer)

    # Verify all arrays are zeroed
    @test all(iszero, roll_buffer.observations)
    @test all(iszero, roll_buffer.actions)
    @test all(iszero, roll_buffer.rewards)
    @test all(iszero, roll_buffer.advantages)
    @test all(iszero, roll_buffer.returns)
    @test all(iszero, roll_buffer.logprobs)
    @test all(iszero, roll_buffer.values)
end

@testitem "Buffer trajectory bootstrap handling" tags = [:buffers, :bootstrap] setup = [SharedTestSetup] begin
    using Random

    # Test bootstrap value handling for truncated episodes
    max_steps = 6
    gamma = 0.9f0
    gae_lambda = 0.8f0
    constant_value = 0.7f0
    bootstrap_value = 0.2f0

    # Create a simple trajectory manually to test bootstrap handling
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    act_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

    # Create trajectory with known values
    traj = Trajectory(obs_space, act_space)

    # Add some dummy data
    for i in 1:max_steps
        push!(traj.observations, rand(Float32, 2))
        push!(traj.actions, rand(Float32, 2))
        push!(traj.rewards, i == max_steps ? 1.0f0 : 0.0f0)  # Reward only at end
        push!(traj.logprobs, 0.0f0)
        push!(traj.values, constant_value)
    end

    # Test terminated trajectory (no bootstrap)
    traj.terminated = true
    traj.truncated = false
    traj.bootstrap_value = nothing

    advantages_terminated = zeros(Float32, max_steps)
    DRiL.compute_advantages!(advantages_terminated, traj, gamma, gae_lambda)

    expected_terminated = SharedTestSetup.compute_expected_gae(
        traj.rewards, traj.values, gamma, gae_lambda; is_terminated = true
    )
    @test isapprox(advantages_terminated, expected_terminated, atol = 1.0e-4)

    # Test truncated trajectory (with bootstrap)
    traj.terminated = false
    traj.truncated = true
    traj.bootstrap_value = bootstrap_value

    advantages_truncated = zeros(Float32, max_steps)
    DRiL.compute_advantages!(advantages_truncated, traj, gamma, gae_lambda)

    expected_truncated = SharedTestSetup.compute_expected_gae(
        traj.rewards, traj.values, gamma, gae_lambda;
        is_terminated = false, bootstrap_value = bootstrap_value
    )
    @test isapprox(advantages_truncated, expected_truncated, atol = 1.0e-4)

    # Verify that bootstrapped case gives different results
    @test !isapprox(advantages_terminated, advantages_truncated, atol = 1.0e-3)
end

@testitem "Buffer data integrity" tags = [:buffers, :integrity] setup = [SharedTestSetup] begin
    using Random

    # Test that buffer maintains data integrity during rollout collection
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])  # Match SimpleRewardEnv shape
    act_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

    n_steps = 16
    n_envs = 2
    gamma = 0.99f0
    gae_lambda = 0.95f0

    roll_buffer = RolloutBuffer(obs_space, act_space, gae_lambda, gamma, n_steps, n_envs)

    # Create simple test environment
    env = MultiThreadedParallelEnv([SharedTestSetup.SimpleRewardEnv(8) for _ in 1:n_envs])
    env_obs_space = DRiL.observation_space(env)
    env_act_space = DRiL.action_space(env)
    @test isequal(env_obs_space, obs_space)
    @test isequal(env_act_space, act_space)


    policy = SharedTestSetup.ConstantValuePolicy(env_obs_space, env_act_space, 0.5f0)
    alg = PPO(n_steps = n_steps, batch_size = 16, epochs = 1)
    agent = Agent(policy, alg; verbose = 0)

    # Collect rollouts
    DRiL.collect_rollout!(roll_buffer, agent, alg, env)

    # Verify buffer dimensions
    @test size(roll_buffer.observations) == (obs_space.shape..., n_steps * n_envs)
    @test size(roll_buffer.actions) == (act_space.shape..., n_steps * n_envs)
    @test length(roll_buffer.rewards) == n_steps * n_envs
    @test length(roll_buffer.advantages) == n_steps * n_envs
    @test length(roll_buffer.returns) == n_steps * n_envs
    @test length(roll_buffer.logprobs) == n_steps * n_envs
    @test length(roll_buffer.values) == n_steps * n_envs

    # Verify no NaN or Inf values
    @test all(isfinite, roll_buffer.rewards)
    @test all(isfinite, roll_buffer.advantages)
    @test all(isfinite, roll_buffer.returns)
    @test all(isfinite, roll_buffer.logprobs)
    @test all(isfinite, roll_buffer.values)

    # Verify returns = advantages + values relationship
    @test isapprox(roll_buffer.returns, roll_buffer.advantages .+ roll_buffer.values, atol = 1.0e-5)
end

@testitem "RolloutBuffer with discrete actions" tags = [:buffers, :rollouts, :discrete] setup = [SharedTestSetup] begin
    using ClassicControlEnvironments
    using Random

    # Test: RolloutBuffer works correctly with discrete action spaces
    cartpole_env() = CartPoleEnv()
    env = MultiThreadedParallelEnv([cartpole_env() for _ in 1:4])
    policy = DiscreteActorCriticLayer(DRiL.observation_space(env), DRiL.action_space(env))
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 1)
    agent = Agent(policy, alg; verbose = 0)

    n_steps = alg.n_steps
    n_envs = DRiL.number_of_envs(env)
    roll_buffer = RolloutBuffer(DRiL.observation_space(env), DRiL.action_space(env), alg.gae_lambda, alg.gamma, n_steps, n_envs)

    # Test rollout collection
    DRiL.collect_rollout!(roll_buffer, agent, alg, env)

    # Check that actions are stored as integer actions in a 1 x total_steps tensor
    actions = roll_buffer.actions
    @test size(actions) == (1, n_steps * n_envs)
    @test all(a -> a ∈ DRiL.action_space(env), vec(actions))
    @test eltype(actions) <: Integer

    # Check observations are valid
    obs = roll_buffer.observations
    obs_space = DRiL.observation_space(env)
    @test size(obs) == (obs_space.shape..., n_steps * n_envs)
    @test eltype(obs) == Float32

    # Check that rewards are reasonable
    rewards = roll_buffer.rewards
    @test all(rewards .>= 0.0f0)  # CartPole gives positive rewards
    @test size(rewards) == (n_steps * n_envs,)

    # Check that log probabilities are consistent
    logprobs = roll_buffer.logprobs
    values = roll_buffer.values
    @test size(logprobs) == (n_steps * n_envs,)
    @test size(values) == (n_steps * n_envs,)

    # Test action evaluation consistency
    ps = agent.train_state.parameters
    st = agent.train_state.states
    onehot_actions = DRiL.discrete_to_onehotbatch(actions, DRiL.action_space(env))
    eval_values, eval_logprobs, entropy, _ = DRiL.evaluate_actions(policy, obs, onehot_actions, ps, st)

    @test isapprox(vec(values), vec(eval_values); atol = 1.0e-5, rtol = 1.0e-5)
    @test isapprox(vec(logprobs), vec(eval_logprobs); atol = 1.0e-5, rtol = 1.0e-5)
    @test all(entropy .>= 0.0f0)  # Entropy should be non-negative
end

@testitem "Discrete vs continuous buffer comparison" tags = [:buffers, :rollouts, :comparison] setup = [SharedTestSetup] begin
    using ClassicControlEnvironments
    using Random

    # Test: Compare buffer behavior between discrete and continuous action spaces

    alg = PPO(n_steps = 4, batch_size = 4, epochs = 1)
    # Discrete environment (CartPole)
    discrete_env = MultiThreadedParallelEnv([CartPoleEnv() for _ in 1:2])
    discrete_policy = DiscreteActorCriticLayer(DRiL.observation_space(discrete_env), DRiL.action_space(discrete_env))
    discrete_agent = Agent(discrete_policy, alg; verbose = 0)

    # Continuous environment (Pendulum)
    continuous_env = MultiThreadedParallelEnv([PendulumEnv() for _ in 1:2])
    continuous_policy = ContinuousActorCriticLayer(DRiL.observation_space(continuous_env), DRiL.action_space(continuous_env))
    continuous_agent = Agent(continuous_policy, alg; verbose = 0)


    # Create buffers
    discrete_buffer = RolloutBuffer(DRiL.observation_space(discrete_env), DRiL.action_space(discrete_env), alg.gae_lambda, alg.gamma, 4, 2)
    continuous_buffer = RolloutBuffer(DRiL.observation_space(continuous_env), DRiL.action_space(continuous_env), alg.gae_lambda, alg.gamma, 4, 2)

    # Collect rollouts
    DRiL.collect_rollout!(discrete_buffer, discrete_agent, alg, discrete_env)
    DRiL.collect_rollout!(continuous_buffer, continuous_agent, alg, continuous_env)

    # Test discrete actions are stored as integers
    discrete_actions = discrete_buffer.actions
    @test eltype(discrete_actions) <: Integer
    @test all(a -> a ∈ DRiL.action_space(discrete_env), vec(discrete_actions))

    # Test continuous actions are floats
    continuous_actions = continuous_buffer.actions
    @test eltype(continuous_actions) <: AbstractFloat
    @test size(continuous_actions) == (1, 4 * 2)  # (action_dim, n_steps*n_envs)

    # Test that both have same buffer structure otherwise
    @test size(discrete_buffer.rewards) == size(continuous_buffer.rewards)
    @test size(discrete_buffer.logprobs) == size(continuous_buffer.logprobs)
    @test size(discrete_buffer.values) == size(continuous_buffer.values)

    # Test that evaluation works for both
    discrete_ps = discrete_agent.train_state.parameters
    discrete_st = discrete_agent.train_state.states
    continuous_ps = continuous_agent.train_state.parameters
    continuous_st = continuous_agent.train_state.states

    # Discrete evaluation
    discrete_onehot_actions = DRiL.discrete_to_onehotbatch(discrete_buffer.actions, DRiL.action_space(discrete_env))
    discrete_eval_values, discrete_eval_logprobs, discrete_entropy, _ = DRiL.evaluate_actions(
        discrete_policy, discrete_buffer.observations, discrete_onehot_actions, discrete_ps, discrete_st
    )

    # Continuous evaluation
    continuous_eval_values, continuous_eval_logprobs, continuous_entropy, _ = DRiL.evaluate_actions(
        continuous_policy, continuous_buffer.observations, continuous_buffer.actions, continuous_ps, continuous_st
    )

    # Test evaluation consistency
    @test isapprox(vec(discrete_buffer.values), vec(discrete_eval_values); atol = 1.0e-5, rtol = 1.0e-5)
    @test isapprox(vec(continuous_buffer.values), vec(continuous_eval_values); atol = 1.0e-5, rtol = 1.0e-5)
    @test isapprox(vec(discrete_buffer.logprobs), vec(discrete_eval_logprobs); atol = 1.0e-5, rtol = 1.0e-5)
    @test isapprox(vec(continuous_buffer.logprobs), vec(continuous_eval_logprobs); atol = 1.0e-5, rtol = 1.0e-5)
end

@testitem "RolloutBuffer with different box shapes" tags = [:buffers, :rollouts] setup = [SharedTestSetup] begin
    using Random
    n_steps = 8

    function get_rollout(env::AbstractEnv)
        obs_space = DRiL.observation_space(env)
        act_space = DRiL.action_space(env)
        #TODO: use a simpler random agent/policy instead?
        policy = ActorCriticLayer(obs_space, act_space)
        alg = PPO()
        agent = Agent(policy, alg; verbose = 0)
        roll_buffer = RolloutBuffer(obs_space, act_space, alg.gae_lambda, alg.gamma, n_steps, DRiL.number_of_envs(env))
        DRiL.collect_rollout!(roll_buffer, agent, alg, env)
        return roll_buffer
    end
    function test_rollout(roll_buffer::RolloutBuffer, env::AbstractEnv)
        act_space = DRiL.action_space(env)
        obs_space = DRiL.observation_space(env)
        act_shape = size(act_space)
        obs_shape = size(obs_space)
        n_envs = DRiL.number_of_envs(env)
        @test size(roll_buffer.observations) == (obs_shape..., n_steps * n_envs)
        @test size(roll_buffer.actions) == (act_shape..., n_steps * n_envs)
        @test length(roll_buffer.rewards) == n_steps * n_envs
        @test length(roll_buffer.advantages) == n_steps * n_envs
        @test length(roll_buffer.returns) == n_steps * n_envs
        @test length(roll_buffer.logprobs) == n_steps * n_envs
    end
    shapes = [(1,), (1, 1), (2,), (2, 3), (2, 3, 1), (2, 3, 4)]
    for shape in shapes
        env = BroadcastedParallelEnv([SharedTestSetup.CustomShapedBoxEnv(shape) for _ in 1:2])
        roll_buffer = get_rollout(env)
        test_rollout(roll_buffer, env)
    end
end

@testitem "Basic ReplayBuffer workings" tags = [:buffers, :rollouts] setup = [SharedTestSetup] begin
    using Random
    using DRiL.DataStructures

    n_envs = 4
    train_freq = 8
    n_steps = floor(Int, train_freq / n_envs)
    buffer_capacity = 16
    rng = Random.Xoshiro(42)

    alg = SAC()
    env = BroadcastedParallelEnv([SharedTestSetup.SimpleRewardEnv(8) for _ in 1:n_envs])
    policy = ContinuousActorCriticLayer(DRiL.observation_space(env), DRiL.action_space(env), critic_type = QCritic())
    agent = Agent(policy, alg)
    buffer = ReplayBuffer(DRiL.observation_space(env), DRiL.action_space(env), buffer_capacity)
    @test capacity(buffer) == buffer_capacity
    @test !isfull(buffer)

    DRiL.collect_rollout!(buffer, agent, alg, env, n_steps)
    @test size(buffer) == n_steps * n_envs

    DRiL.collect_rollout!(buffer, agent, alg, env, train_freq)
    @test size(buffer) == buffer_capacity
    @test isfull(buffer)

    empty!(buffer)
    @test size(buffer) == 0
    @test isempty(buffer)
end
