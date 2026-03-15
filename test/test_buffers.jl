using Test
using Drill
using Random
using ClassicControlEnvironments
using Statistics
include("setup.jl")
using .TestSetup

@testset "Buffer logprobs consistency" begin
    pend_env() = PendulumEnv()
    env = MultiThreadedParallelEnv([pend_env() for _ in 1:4])
    layer = ActorCriticLayer(Drill.observation_space(env), Drill.action_space(env))
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 1)
    agent = Agent(layer, alg; verbose = 0)
    n_steps = alg.n_steps
    n_envs = Drill.number_of_envs(env)
    roll_buffer = RolloutBuffer(Drill.observation_space(env), Drill.action_space(env), alg.gae_lambda, alg.gamma, n_steps, n_envs)

    for i in 1:10
        Drill.collect_rollout!(roll_buffer, agent, alg, env)
        obs = roll_buffer.observations
        act = roll_buffer.actions
        logprobs = roll_buffer.logprobs
        ps = agent.train_state.parameters
        st = agent.train_state.states
        _, new_logprobs, _, _ = Drill.evaluate_actions(layer, obs, act, ps, st)
        @test isapprox(vec(logprobs), vec(new_logprobs); atol = 1.0e-5, rtol = 1.0e-5)
    end
end

@testset "Buffer reset functionality" begin
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    act_space = Box(Float32[-1.0], Float32[1.0])

    roll_buffer = RolloutBuffer(obs_space, act_space, 0.95f0, 0.99f0, 8, 2)

    roll_buffer.observations .= 1.0f0
    roll_buffer.actions .= 2.0f0
    roll_buffer.rewards .= 3.0f0
    roll_buffer.advantages .= 4.0f0
    roll_buffer.returns .= 5.0f0
    roll_buffer.logprobs .= 6.0f0
    roll_buffer.values .= 7.0f0

    Drill.reset!(roll_buffer)

    @test all(iszero, roll_buffer.observations)
    @test all(iszero, roll_buffer.actions)
    @test all(iszero, roll_buffer.rewards)
    @test all(iszero, roll_buffer.advantages)
    @test all(iszero, roll_buffer.returns)
    @test all(iszero, roll_buffer.logprobs)
    @test all(iszero, roll_buffer.values)
end

@testset "Buffer trajectory bootstrap handling" begin
    max_steps = 6
    gamma = 0.9f0
    gae_lambda = 0.8f0
    constant_value = 0.7f0
    bootstrap_value = 0.2f0

    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    act_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

    traj = Trajectory(obs_space, act_space)

    for i in 1:max_steps
        push!(traj.observations, rand(Float32, 2))
        push!(traj.actions, rand(Float32, 2))
        push!(traj.rewards, i == max_steps ? 1.0f0 : 0.0f0)
        push!(traj.logprobs, 0.0f0)
        push!(traj.values, constant_value)
    end

    traj.terminated = true
    traj.truncated = false
    traj.bootstrap_value = nothing

    advantages_terminated = zeros(Float32, max_steps)
    Drill.compute_advantages!(advantages_terminated, traj, gamma, gae_lambda)

    expected_terminated = compute_expected_gae(
        traj.rewards, traj.values, gamma, gae_lambda; is_terminated = true
    )
    @test isapprox(advantages_terminated, expected_terminated, atol = 1.0e-4)

    traj.terminated = false
    traj.truncated = true
    traj.bootstrap_value = bootstrap_value

    advantages_truncated = zeros(Float32, max_steps)
    Drill.compute_advantages!(advantages_truncated, traj, gamma, gae_lambda)

    expected_truncated = compute_expected_gae(
        traj.rewards, traj.values, gamma, gae_lambda;
        is_terminated = false, bootstrap_value = bootstrap_value
    )
    @test isapprox(advantages_truncated, expected_truncated, atol = 1.0e-4)

    @test !isapprox(advantages_terminated, advantages_truncated, atol = 1.0e-3)
end

@testset "Buffer data integrity" begin
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    act_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

    n_steps = 16
    n_envs = 2
    gamma = 0.99f0
    gae_lambda = 0.95f0

    roll_buffer = RolloutBuffer(obs_space, act_space, gae_lambda, gamma, n_steps, n_envs)

    env = MultiThreadedParallelEnv([SimpleRewardEnv(8) for _ in 1:n_envs])
    env_obs_space = Drill.observation_space(env)
    env_act_space = Drill.action_space(env)
    @test isequal(env_obs_space, obs_space)
    @test isequal(env_act_space, act_space)

    layer = ConstantValueLayer(env_obs_space, env_act_space, 0.5f0)
    alg = PPO(n_steps = n_steps, batch_size = 16, epochs = 1)
    agent = Agent(layer, alg; verbose = 0)

    Drill.collect_rollout!(roll_buffer, agent, alg, env)

    @test size(roll_buffer.observations) == (obs_space.shape..., n_steps * n_envs)
    @test size(roll_buffer.actions) == (act_space.shape..., n_steps * n_envs)
    @test length(roll_buffer.rewards) == n_steps * n_envs
    @test length(roll_buffer.advantages) == n_steps * n_envs
    @test length(roll_buffer.returns) == n_steps * n_envs
    @test length(roll_buffer.logprobs) == n_steps * n_envs
    @test length(roll_buffer.values) == n_steps * n_envs

    @test all(isfinite, roll_buffer.rewards)
    @test all(isfinite, roll_buffer.advantages)
    @test all(isfinite, roll_buffer.returns)
    @test all(isfinite, roll_buffer.logprobs)
    @test all(isfinite, roll_buffer.values)

    @test isapprox(roll_buffer.returns, roll_buffer.advantages .+ roll_buffer.values, atol = 1.0e-5)
end

@testset "RolloutBuffer with discrete actions" begin
    cartpole_env() = CartPoleEnv()
    env = MultiThreadedParallelEnv([cartpole_env() for _ in 1:4])
    layer = DiscreteActorCriticLayer(Drill.observation_space(env), Drill.action_space(env))
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 1)
    agent = Agent(layer, alg; verbose = 0)

    n_steps = alg.n_steps
    n_envs = Drill.number_of_envs(env)
    roll_buffer = RolloutBuffer(Drill.observation_space(env), Drill.action_space(env), alg.gae_lambda, alg.gamma, n_steps, n_envs)

    Drill.collect_rollout!(roll_buffer, agent, alg, env)

    actions = roll_buffer.actions
    @test size(actions) == (1, n_steps * n_envs)
    @test all(a -> a ∈ Drill.action_space(env), vec(actions))
    @test eltype(actions) <: Integer

    obs = roll_buffer.observations
    obs_space = Drill.observation_space(env)
    @test size(obs) == (obs_space.shape..., n_steps * n_envs)
    @test eltype(obs) == Float32

    rewards = roll_buffer.rewards
    @test all(rewards .>= 0.0f0)
    @test size(rewards) == (n_steps * n_envs,)

    logprobs = roll_buffer.logprobs
    values = roll_buffer.values
    @test size(logprobs) == (n_steps * n_envs,)
    @test size(values) == (n_steps * n_envs,)

    ps = agent.train_state.parameters
    st = agent.train_state.states
    onehot_actions = Drill.discrete_to_onehotbatch(actions, Drill.action_space(env))
    eval_values, eval_logprobs, entropy, _ = Drill.evaluate_actions(layer, obs, onehot_actions, ps, st)

    @test isapprox(vec(values), vec(eval_values); atol = 1.0e-5, rtol = 1.0e-5)
    @test isapprox(vec(logprobs), vec(eval_logprobs); atol = 1.0e-5, rtol = 1.0e-5)
    @test all(entropy .>= 0.0f0)
end

@testset "Discrete vs continuous buffer comparison" begin
    alg = PPO(n_steps = 4, batch_size = 4, epochs = 1)
    discrete_env = MultiThreadedParallelEnv([CartPoleEnv() for _ in 1:2])
    discrete_layer = DiscreteActorCriticLayer(Drill.observation_space(discrete_env), Drill.action_space(discrete_env))
    discrete_agent = Agent(discrete_layer, alg; verbose = 0)

    continuous_env = MultiThreadedParallelEnv([PendulumEnv() for _ in 1:2])
    continuous_layer = ContinuousActorCriticLayer(Drill.observation_space(continuous_env), Drill.action_space(continuous_env))
    continuous_agent = Agent(continuous_layer, alg; verbose = 0)

    discrete_buffer = RolloutBuffer(Drill.observation_space(discrete_env), Drill.action_space(discrete_env), alg.gae_lambda, alg.gamma, 4, 2)
    continuous_buffer = RolloutBuffer(Drill.observation_space(continuous_env), Drill.action_space(continuous_env), alg.gae_lambda, alg.gamma, 4, 2)

    Drill.collect_rollout!(discrete_buffer, discrete_agent, alg, discrete_env)
    Drill.collect_rollout!(continuous_buffer, continuous_agent, alg, continuous_env)

    discrete_actions = discrete_buffer.actions
    @test eltype(discrete_actions) <: Integer
    @test all(a -> a ∈ Drill.action_space(discrete_env), vec(discrete_actions))

    continuous_actions = continuous_buffer.actions
    @test eltype(continuous_actions) <: AbstractFloat
    @test size(continuous_actions) == (1, 4 * 2)

    @test size(discrete_buffer.rewards) == size(continuous_buffer.rewards)
    @test size(discrete_buffer.logprobs) == size(continuous_buffer.logprobs)
    @test size(discrete_buffer.values) == size(continuous_buffer.values)

    discrete_ps = discrete_agent.train_state.parameters
    discrete_st = discrete_agent.train_state.states
    continuous_ps = continuous_agent.train_state.parameters
    continuous_st = continuous_agent.train_state.states

    discrete_onehot_actions = Drill.discrete_to_onehotbatch(discrete_buffer.actions, Drill.action_space(discrete_env))
    discrete_eval_values, discrete_eval_logprobs, discrete_entropy, _ = Drill.evaluate_actions(
        discrete_layer, discrete_buffer.observations, discrete_onehot_actions, discrete_ps, discrete_st
    )

    continuous_eval_values, continuous_eval_logprobs, continuous_entropy, _ = Drill.evaluate_actions(
        continuous_layer, continuous_buffer.observations, continuous_buffer.actions, continuous_ps, continuous_st
    )

    @test isapprox(vec(discrete_buffer.values), vec(discrete_eval_values); atol = 1.0e-5, rtol = 1.0e-5)
    @test isapprox(vec(continuous_buffer.values), vec(continuous_eval_values); atol = 1.0e-5, rtol = 1.0e-5)
    @test isapprox(vec(discrete_buffer.logprobs), vec(discrete_eval_logprobs); atol = 1.0e-5, rtol = 1.0e-5)
    @test isapprox(vec(continuous_buffer.logprobs), vec(continuous_eval_logprobs); atol = 1.0e-5, rtol = 1.0e-5)
end

@testset "RolloutBuffer with different box shapes" begin
    n_steps = 8

    function get_rollout(env::AbstractEnv)
        obs_space = Drill.observation_space(env)
        act_space = Drill.action_space(env)
        layer = ActorCriticLayer(obs_space, act_space)
        alg = PPO()
        agent = Agent(layer, alg; verbose = 0)
        roll_buffer = RolloutBuffer(obs_space, act_space, alg.gae_lambda, alg.gamma, n_steps, Drill.number_of_envs(env))
        Drill.collect_rollout!(roll_buffer, agent, alg, env)
        return roll_buffer
    end

    function test_rollout(roll_buffer::RolloutBuffer, env::AbstractEnv)
        act_space = Drill.action_space(env)
        obs_space = Drill.observation_space(env)
        act_shape = size(act_space)
        obs_shape = size(obs_space)
        n_envs = Drill.number_of_envs(env)
        @test size(roll_buffer.observations) == (obs_shape..., n_steps * n_envs)
        @test size(roll_buffer.actions) == (act_shape..., n_steps * n_envs)
        @test length(roll_buffer.rewards) == n_steps * n_envs
        @test length(roll_buffer.advantages) == n_steps * n_envs
        @test length(roll_buffer.returns) == n_steps * n_envs
        @test length(roll_buffer.logprobs) == n_steps * n_envs
    end

    shapes = [(1,), (1, 1), (2,), (2, 3), (2, 3, 1), (2, 3, 4)]
    for shape in shapes
        env = BroadcastedParallelEnv([CustomShapedBoxEnv(shape) for _ in 1:2])
        roll_buffer = get_rollout(env)
        test_rollout(roll_buffer, env)
    end
end

@testset "Basic ReplayBuffer workings" begin
    using Drill.DataStructures

    n_envs = 4
    train_freq = 8
    n_steps = floor(Int, train_freq / n_envs)
    buffer_capacity = 16
    rng = Random.Xoshiro(42)

    alg = SAC()
    env = BroadcastedParallelEnv([SimpleRewardEnv(8) for _ in 1:n_envs])
    layer = ContinuousActorCriticLayer(Drill.observation_space(env), Drill.action_space(env), critic_type = QCritic())
    agent = Agent(layer, alg)
    buffer = ReplayBuffer(Drill.observation_space(env), Drill.action_space(env), buffer_capacity)
    @test capacity(buffer) == buffer_capacity
    @test !isfull(buffer)

    Drill.collect_rollout!(buffer, agent, alg, env, n_steps)
    @test size(buffer) == n_steps * n_envs

    Drill.collect_rollout!(buffer, agent, alg, env, train_freq)
    @test size(buffer) == buffer_capacity
    @test isfull(buffer)

    empty!(buffer)
    @test size(buffer) == 0
    @test isempty(buffer)
end
