using Test
using Drill
using DrillInterface
using Random
using Statistics
using Drill.DataStructures: capacity, isfull
include("setup.jl")
using .TestSetup

# Small discrete-action env: Box(4) observations, two actions,
# reward 1 per step, termination after `max_steps` steps.
mutable struct DiscreteCountingEnv <: AbstractEnv
    max_steps::Int
    steps::Int
    rng::Random.AbstractRNG
end
DiscreteCountingEnv(max_steps::Int = 10) = DiscreteCountingEnv(max_steps, 0, Random.Xoshiro())

DrillInterface.observation_space(::DiscreteCountingEnv) = Box(-1.0f0, 1.0f0, (4,))
DrillInterface.action_space(::DiscreteCountingEnv) = Discrete(2)
DrillInterface.observe(env::DiscreteCountingEnv) = rand(env.rng, Float32, 4) .* 2.0f0 .- 1.0f0
DrillInterface.terminated(env::DiscreteCountingEnv) = env.steps >= env.max_steps
DrillInterface.truncated(::DiscreteCountingEnv) = false
function DrillInterface.act!(env::DiscreteCountingEnv, action::Integer)
    @assert action in DrillInterface.action_space(env)
    env.steps += 1
    return 1.0f0
end
function DrillInterface.reset!(env::DiscreteCountingEnv; seed = nothing)
    isnothing(seed) || Random.seed!(env.rng, seed)
    env.steps = 0
    return nothing
end

# Deterministic env: observation `[steps, id]`, reward `10 * id + steps`. The episode
# terminates after `max_steps` steps and is truncated after `trunc_steps` steps.
mutable struct StepCounterEnv <: AbstractEnv
    id::Int
    max_steps::Int
    trunc_steps::Int
    steps::Int
end
StepCounterEnv(id, max_steps, trunc_steps) = StepCounterEnv(id, max_steps, trunc_steps, 0)

DrillInterface.observation_space(::StepCounterEnv) = Box(Float32[0.0, 0.0], Float32[100.0, 100.0])
DrillInterface.action_space(::StepCounterEnv) = Box(Float32[-1.0], Float32[1.0])
DrillInterface.observe(env::StepCounterEnv) = Float32[env.steps, env.id]
DrillInterface.terminated(env::StepCounterEnv) = env.steps >= env.max_steps
DrillInterface.truncated(env::StepCounterEnv) = !DrillInterface.terminated(env) && env.steps >= env.trunc_steps
DrillInterface.get_info(::StepCounterEnv) = Dict{String, Any}()
function DrillInterface.act!(env::StepCounterEnv, action::AbstractArray)
    env.steps += 1
    return Float32(10 * env.id + env.steps)
end
function DrillInterface.reset!(env::StepCounterEnv; seed = nothing)
    env.steps = 0
    return nothing
end

# Expected per-step data of a `StepCounterEnv(id, max_steps, trunc_steps)` started at step 0:
# observation step counter, reward, terminated, truncated, and step counter after the step.
function expected_counter_steps(id, max_steps, trunc_steps, n_steps)
    steps = 0
    out = NamedTuple[]
    for _ in 1:n_steps
        before = steps
        steps += 1
        term = steps >= max_steps
        trunc = !term && steps >= trunc_steps
        push!(out, (; before, after = steps, reward = Float32(10 * id + steps), term, trunc))
        (term || trunc) && (steps = 0)
    end
    return out
end

function make_cache(env, layer, alg; max_steps = alg.n_steps * DrillInterface.number_of_envs(env))
    return init(RLProblem(env, layer), alg; max_steps, verbosity = 0)
end

function collect_and_prepare!(roll_buffer, cache, alg, env)
    Drill.collect_rollout!(roll_buffer, cache, alg, env)
    Drill.prepare_rollout!(roll_buffer, alg)
    return roll_buffer
end

@testset "Buffer logprobs consistency" begin
    env = MultiThreadedParallelEnv([TrackingTargetEnv() for _ in 1:4])
    layer = ActorCriticModel(DrillInterface.observation_space(env), DrillInterface.action_space(env))
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 1)
    cache = make_cache(env, layer, alg)
    n_steps = alg.n_steps
    n_envs = DrillInterface.number_of_envs(env)
    roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), n_steps, n_envs)

    for i in 1:10
        collect_and_prepare!(roll_buffer, cache, alg, env)
        obs = roll_buffer.observations
        act = roll_buffer.actions
        logprobs = roll_buffer.logprobs
        ps = Drill.parameters(cache)
        st = Drill.states(cache)
        _, new_logprobs, _, _ = Drill.evaluate_actions(layer, obs, act, ps, st)
        @test isapprox(vec(logprobs), vec(new_logprobs); atol = 1.0e-5, rtol = 1.0e-5)
    end
end

@testset "Buffer reset functionality" begin
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    act_space = Box(Float32[-1.0], Float32[1.0])

    roll_buffer = RolloutBuffer(obs_space, act_space, 8, 2)

    roll_buffer.observations .= 1.0f0
    roll_buffer.actions .= 2.0f0
    roll_buffer.rewards .= 3.0f0
    roll_buffer.advantages .= 4.0f0
    roll_buffer.returns .= 5.0f0
    roll_buffer.logprobs .= 6.0f0
    roll_buffer.values .= 7.0f0

    DrillInterface.reset!(roll_buffer)

    @test all(iszero, roll_buffer.observations)
    @test all(iszero, roll_buffer.actions)
    @test all(iszero, roll_buffer.rewards)
    @test all(iszero, roll_buffer.advantages)
    @test all(iszero, roll_buffer.returns)
    @test all(iszero, roll_buffer.logprobs)
    @test all(iszero, roll_buffer.values)
end

@testset "Buffer compatibility traits" begin
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    act_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    roll_buffer = RolloutBuffer(obs_space, act_space, 4, 1)
    replay_buffer = ReplayBuffer(obs_space, act_space, 8)
    ppo = PPO(; n_steps = 4, batch_size = 4, epochs = 1)
    sac = SAC(; buffer_capacity = 8, batch_size = 2, start_steps = 0)

    @test isa(roll_buffer, OnPolicyBuffer)
    @test isa(replay_buffer, OffPolicyBuffer)
    @test compatible(ppo, roll_buffer)
    @test !compatible(ppo, replay_buffer)
    @test compatible(sac, replay_buffer)
    @test !compatible(sac, roll_buffer)

    env = BroadcastedParallelEnv([SimpleRewardEnv(4)])
    ppo_layer = ConstantValueModel(
        DrillInterface.observation_space(env),
        DrillInterface.action_space(env),
        0.5f0,
    )
    sac_layer = SACModel(
        DrillInterface.observation_space(env),
        DrillInterface.action_space(env),
    )

    @test_throws ArgumentError init(
        RLProblem(env, ppo_layer),
        ppo;
        max_steps = 4,
        buffer = replay_buffer,
        verbosity = 0,
    )
    @test_throws ArgumentError init(
        RLProblem(env, sac_layer),
        sac;
        max_steps = 4,
        buffer = roll_buffer,
        verbosity = 0,
    )
end

@testset "store_step! and compute_gae! bootstrap handling" begin
    n_steps = 6
    gamma = 0.9f0
    gae_lambda = 0.8f0
    constant_value = 0.7f0
    bootstrap_value = 0.2f0

    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    act_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    rewards = [i == n_steps ? 1.0f0 : 0.0f0 for i in 1:n_steps]
    values = fill(constant_value, n_steps)

    function filled_buffer(terminated_last::Bool, truncated_last::Bool)
        buffer = RolloutBuffer(obs_space, act_space, n_steps, 1)
        for t in 1:n_steps
            store_step!(
                buffer, t, rand(Float32, 2, 1), rand(Float32, 2, 1),
                [rewards[t]], [0.0f0], [values[t]],
                [t == n_steps && terminated_last], [t == n_steps && truncated_last],
            )
        end
        buffer.bootstrap_values[n_steps] = bootstrap_value
        return buffer
    end

    buffer_terminated = filled_buffer(true, false)
    compute_gae!(buffer_terminated, gamma, gae_lambda)
    expected_terminated = compute_expected_gae(rewards, values, gamma, gae_lambda; is_terminated = true)
    @test isapprox(buffer_terminated.advantages, expected_terminated, atol = 1.0e-4)

    buffer_truncated = filled_buffer(false, true)
    compute_gae!(buffer_truncated, gamma, gae_lambda)
    expected_truncated = compute_expected_gae(
        rewards, values, gamma, gae_lambda;
        is_terminated = false, bootstrap_value = bootstrap_value
    )
    @test isapprox(buffer_truncated.advantages, expected_truncated, atol = 1.0e-4)
    @test !isapprox(buffer_terminated.advantages, buffer_truncated.advantages, atol = 1.0e-3)

    # Rollout end without a done flag bootstraps the same way as truncation.
    buffer_open = filled_buffer(false, false)
    compute_gae!(buffer_open, gamma, gae_lambda)
    @test isapprox(buffer_open.advantages, expected_truncated, atol = 1.0e-4)
end

@testset "store_step! step-major layout" begin
    obs_space = Box(-10.0f0, 10.0f0, (2,))
    act_space = Box(-1.0f0, 1.0f0, (1,))
    n_steps, n_envs = 3, 2
    buffer = RolloutBuffer(obs_space, act_space, n_steps, n_envs)
    @test length(buffer) == n_steps * n_envs
    @test step_indices(buffer, 1) == 1:2
    @test step_indices(buffer, 3) == 5:6
    for t in 1:n_steps
        obs = reshape(Float32[t, -t, 10 + t, -10 - t] ./ 10, 2, 2)
        store_step!(
            buffer, t, obs, reshape(Float32[t, -t] ./ 10, 1, 2),
            Float32[t, 10 + t], Float32[-t, -10 - t], Float32[2t, 20 + 2t],
            [false, t == 2], [t == 3, false],
        )
    end
    @test buffer.rewards == Float32[1, 11, 2, 12, 3, 13]
    @test buffer.logprobs == -buffer.rewards
    @test buffer.values == 2 .* buffer.rewards
    @test buffer.terminateds == [false, false, false, true, false, false]
    @test buffer.truncateds == [false, false, false, false, true, false]
    @test buffer.observations[:, step_indices(buffer, 2)] == reshape(Float32[2, -2, 12, -12] ./ 10, 2, 2)
    @test buffer.actions[:, step_indices(buffer, 3)] == reshape(Float32[3, -3] ./ 10, 1, 2)
end

@testset "collect_rollout! layout, done flags and bootstrap values" begin
    n_steps = 7
    specs = [(1, 3, 100), (2, 100, 4), (3, 100, 100)]
    n_envs = length(specs)
    expected = [expected_counter_steps(id, m, k, n_steps) for (id, m, k) in specs]

    function counter_rollout(layer_fn)
        env = BroadcastedParallelEnv([StepCounterEnv(spec...) for spec in specs])
        obs_space = DrillInterface.observation_space(env)
        act_space = DrillInterface.action_space(env)
        layer = layer_fn(obs_space, act_space)
        alg = PPO(; n_steps, batch_size = n_steps, epochs = 1)
        cache = make_cache(env, layer, alg)
        buffer = RolloutBuffer(obs_space, act_space, n_steps, n_envs)
        _, success = Drill.collect_rollout!(buffer, cache, alg, env)
        @test success
        return buffer, cache
    end

    constant_value = 0.25f0
    buffer, _ = counter_rollout((o, a) -> ConstantValueModel(o, a, constant_value))
    for t in 1:n_steps, e in 1:n_envs
        i = step_indices(buffer, t)[e]
        x = expected[e][t]
        @test buffer.observations[:, i] == Float32[x.before, specs[e][1]]
        @test buffer.rewards[i] == x.reward
        @test buffer.terminateds[i] == x.term
        @test buffer.truncateds[i] == x.trunc
        @test buffer.values[i] == constant_value
        needs_bootstrap = x.trunc || (t == n_steps && !x.term)
        @test buffer.bootstrap_values[i] == (needs_bootstrap ? constant_value : 0.0f0)
    end
    # Env 1 terminates at steps 3 and 6, env 2 is truncated at step 4.
    @test findall(buffer.terminateds) == [step_indices(buffer, 3)[1], step_indices(buffer, 6)[1]]
    @test findall(buffer.truncateds) == [step_indices(buffer, 4)[2]]

    # With an observation-dependent critic, bootstrap values must come from the real next
    # observation: the final observation of a truncated episode, not the reset one.
    buffer, cache = counter_rollout((o, a) -> ActorCriticModel(o, a))
    for t in 1:n_steps, e in 1:n_envs
        i = step_indices(buffer, t)[e]
        x = expected[e][t]
        if x.trunc || (t == n_steps && !x.term)
            next_obs = reshape(Float32[x.after, specs[e][1]], 2, 1)
            @test buffer.bootstrap_values[i] ≈ only(Drill.predict_values(cache, next_obs)) atol = 1.0e-5
        else
            @test buffer.bootstrap_values[i] == 0.0f0
        end
        @test buffer.values[i] ≈ only(Drill.predict_values(cache, buffer.observations[:, i:i])) atol = 1.0e-5
    end
    # Sanity check that the critic distinguishes the final from the reset observation.
    truncated_i = step_indices(buffer, 4)[2]
    reset_value = only(Drill.predict_values(cache, reshape(Float32[0, 2], 2, 1)))
    @test !isapprox(buffer.bootstrap_values[truncated_i], reset_value; atol = 1.0e-6)
end

@testset "Buffer data integrity" begin
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    act_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

    n_steps = 16
    n_envs = 2
    gamma = 0.99f0
    gae_lambda = 0.95f0

    roll_buffer = RolloutBuffer(obs_space, act_space, n_steps, n_envs)

    env = MultiThreadedParallelEnv([SimpleRewardEnv(8) for _ in 1:n_envs])
    env_obs_space = DrillInterface.observation_space(env)
    env_act_space = DrillInterface.action_space(env)
    @test isequal(env_obs_space, obs_space)
    @test isequal(env_act_space, act_space)

    layer = ConstantValueModel(env_obs_space, env_act_space, 0.5f0)
    alg = PPO(n_steps = n_steps, batch_size = 16, epochs = 1)
    cache = make_cache(env, layer, alg)

    collect_and_prepare!(roll_buffer, cache, alg, env)

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
    env = MultiThreadedParallelEnv([DiscreteCountingEnv() for _ in 1:4])
    layer = DiscreteActorCriticModel(DrillInterface.observation_space(env), DrillInterface.action_space(env))
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 1)
    cache = make_cache(env, layer, alg)

    n_steps = alg.n_steps
    n_envs = DrillInterface.number_of_envs(env)
    roll_buffer = RolloutBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), n_steps, n_envs)

    collect_and_prepare!(roll_buffer, cache, alg, env)

    actions = roll_buffer.actions
    @test size(actions) == (1, n_steps * n_envs)
    @test all(a -> a ∈ DrillInterface.action_space(env), vec(actions))
    @test eltype(actions) <: Integer

    obs = roll_buffer.observations
    obs_space = DrillInterface.observation_space(env)
    @test size(obs) == (obs_space.shape..., n_steps * n_envs)
    @test eltype(obs) == Float32

    rewards = roll_buffer.rewards
    @test all(rewards .>= 0.0f0)
    @test size(rewards) == (n_steps * n_envs,)

    logprobs = roll_buffer.logprobs
    values = roll_buffer.values
    @test size(logprobs) == (n_steps * n_envs,)
    @test size(values) == (n_steps * n_envs,)

    ps = Drill.parameters(cache)
    st = Drill.states(cache)
    onehot_actions = Drill.discrete_to_onehotbatch(actions, DrillInterface.action_space(env))
    eval_values, eval_logprobs, entropy, _ = Drill.evaluate_actions(layer, obs, onehot_actions, ps, st)

    @test isapprox(vec(values), vec(eval_values); atol = 1.0e-5, rtol = 1.0e-5)
    @test isapprox(vec(logprobs), vec(eval_logprobs); atol = 1.0e-5, rtol = 1.0e-5)
    @test all(entropy .>= 0.0f0)
end

@testset "Discrete vs continuous buffer comparison" begin
    alg = PPO(n_steps = 4, batch_size = 4, epochs = 1)
    discrete_env = MultiThreadedParallelEnv([DiscreteCountingEnv() for _ in 1:2])
    discrete_layer = DiscreteActorCriticModel(DrillInterface.observation_space(discrete_env), DrillInterface.action_space(discrete_env))
    discrete_cache = make_cache(discrete_env, discrete_layer, alg)

    continuous_env = MultiThreadedParallelEnv([TrackingTargetEnv() for _ in 1:2])
    continuous_layer = ContinuousActorCriticModel(DrillInterface.observation_space(continuous_env), DrillInterface.action_space(continuous_env))
    continuous_cache = make_cache(continuous_env, continuous_layer, alg)

    discrete_buffer = RolloutBuffer(DrillInterface.observation_space(discrete_env), DrillInterface.action_space(discrete_env), 4, 2)
    continuous_buffer = RolloutBuffer(DrillInterface.observation_space(continuous_env), DrillInterface.action_space(continuous_env), 4, 2)

    collect_and_prepare!(discrete_buffer, discrete_cache, alg, discrete_env)
    collect_and_prepare!(continuous_buffer, continuous_cache, alg, continuous_env)

    discrete_actions = discrete_buffer.actions
    @test eltype(discrete_actions) <: Integer
    @test all(a -> a ∈ DrillInterface.action_space(discrete_env), vec(discrete_actions))

    continuous_actions = continuous_buffer.actions
    @test eltype(continuous_actions) <: AbstractFloat
    @test size(continuous_actions) == (1, 4 * 2)

    @test size(discrete_buffer.rewards) == size(continuous_buffer.rewards)
    @test size(discrete_buffer.logprobs) == size(continuous_buffer.logprobs)
    @test size(discrete_buffer.values) == size(continuous_buffer.values)

    discrete_ps = Drill.parameters(discrete_cache)
    discrete_st = Drill.states(discrete_cache)
    continuous_ps = Drill.parameters(continuous_cache)
    continuous_st = Drill.states(continuous_cache)

    discrete_onehot_actions = Drill.discrete_to_onehotbatch(discrete_buffer.actions, DrillInterface.action_space(discrete_env))
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

    function get_rollout(env::AbstractParallelEnv)
        obs_space = DrillInterface.observation_space(env)
        act_space = DrillInterface.action_space(env)
        layer = ActorCriticModel(obs_space, act_space)
        alg = PPO()
        cache = make_cache(env, layer, alg)
        roll_buffer = RolloutBuffer(obs_space, act_space, n_steps, DrillInterface.number_of_envs(env))
        collect_and_prepare!(roll_buffer, cache, alg, env)
        return roll_buffer
    end

    function test_rollout(roll_buffer::RolloutBuffer, env::AbstractParallelEnv)
        act_space = DrillInterface.action_space(env)
        obs_space = DrillInterface.observation_space(env)
        act_shape = size(act_space)
        obs_shape = size(obs_space)
        n_envs = DrillInterface.number_of_envs(env)
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


    n_envs = 4
    train_freq = 8
    n_steps = floor(Int, train_freq / n_envs)
    buffer_capacity = 16
    rng = Random.Xoshiro(42)

    alg = SAC()
    env = BroadcastedParallelEnv([SimpleRewardEnv(8) for _ in 1:n_envs])
    layer = ContinuousActorCriticModel(DrillInterface.observation_space(env), DrillInterface.action_space(env), critic_type = QCritic())
    cache = init(RLProblem(env, layer), alg; max_steps = train_freq * 4, verbosity = 0, rng)
    buffer = ReplayBuffer(DrillInterface.observation_space(env), DrillInterface.action_space(env), buffer_capacity)
    @test capacity(buffer) == buffer_capacity
    @test !isfull(buffer)

    Drill.collect_rollout!(buffer, cache, alg, env, n_steps)
    @test size(buffer) == n_steps * n_envs

    Drill.collect_rollout!(buffer, cache, alg, env, train_freq)
    @test size(buffer) == buffer_capacity
    @test isfull(buffer)

    empty!(buffer)
    @test size(buffer) == 0
    @test isempty(buffer)
end

@testset "ReplayBuffer add_transitions! and ring wrap-around" begin
    obs_space = Box(-100.0f0, 100.0f0, (2,))
    act_space = Box(-100.0f0, 100.0f0, (1,))
    buffer = ReplayBuffer(obs_space, act_space, 5)
    @test capacity(buffer) == 5
    @test isempty(buffer)
    @test !isfull(buffer)
    @test length(buffer) == 0

    # Transition k has observation [k, -k], action k, reward k and next observation [k + 1, -k].
    function add_range!(buffer, ks)
        n = length(ks)
        obs = Float32.(vcat(ks', -ks'))
        next_obs = Float32.(vcat(ks' .+ 1, -ks'))
        add_transitions!(
            buffer, obs, Float32.(reshape(ks, 1, n)), Float32.(ks),
            iseven.(ks), ks .% 3 .== 0, next_obs,
        )
        return buffer
    end

    add_range!(buffer, 1:3)
    @test length(buffer) == 3
    @test !isempty(buffer)
    @test !isfull(buffer)
    @test buffer.rewards[1:3] == Float32[1, 2, 3]
    @test buffer.observations[:, 1:3] == Float32[1 2 3; -1 -2 -3]
    @test buffer.next_observations[:, 1:3] == Float32[2 3 4; -1 -2 -3]
    @test buffer.terminated[1:3] == [false, true, false]
    @test buffer.truncated[1:3] == [false, false, true]

    add_range!(buffer, 4:7)
    @test length(buffer) == 5
    @test isfull(buffer)
    @test buffer.position == 3
    # Transitions 6 and 7 overwrote the two oldest entries.
    @test buffer.rewards == Float32[6, 7, 3, 4, 5]
    @test buffer.actions == Float32[6 7 3 4 5]
    @test buffer.observations[1, :] == Float32[6, 7, 3, 4, 5]
    @test buffer.next_observations[1, :] == Float32[7, 8, 4, 5, 6]
    @test buffer.terminated == iseven.([6, 7, 3, 4, 5])
    @test buffer.truncated == ([6, 7, 3, 4, 5] .% 3 .== 0)

    add_range!(buffer, 8:20)
    @test length(buffer) == 5
    @test sort(buffer.rewards) == Float32[16, 17, 18, 19, 20]

    empty!(buffer)
    @test isempty(buffer)
    @test length(buffer) == 0
    @test !isfull(buffer)
    add_range!(buffer, 1:1)
    @test length(buffer) == 1
    @test buffer.rewards[1] == 1.0f0
end

@testset "ReplayBuffer collect_rollout! stores true next observations" begin
    n_steps = 6
    specs = [(1, 2, 100), (2, 100, 3)]
    n_envs = length(specs)
    env = BroadcastedParallelEnv([StepCounterEnv(spec...) for spec in specs])
    obs_space = DrillInterface.observation_space(env)
    act_space = DrillInterface.action_space(env)
    layer = SACModel(obs_space, act_space)
    alg = SAC(; buffer_capacity = 100, batch_size = 4, start_steps = 0)
    cache = init(RLProblem(env, layer), alg; max_steps = 64, verbosity = 0, rng = Random.Xoshiro(1))
    buffer = ReplayBuffer(obs_space, act_space, 100)

    _, success = Drill.collect_rollout!(buffer, cache, alg, env, n_steps)
    @test success
    @test length(buffer) == n_steps * n_envs
    expected = [expected_counter_steps(id, m, k, n_steps) for (id, m, k) in specs]
    for t in 1:n_steps, e in 1:n_envs
        i = (t - 1) * n_envs + e
        x = expected[e][t]
        id = specs[e][1]
        @test buffer.observations[:, i] == Float32[x.before, id]
        # Mid-episode this is the next observation; for a finished episode it is the
        # final observation before the reset.
        @test buffer.next_observations[:, i] == Float32[x.after, id]
        @test buffer.rewards[i] == x.reward
        @test buffer.terminated[i] == x.term
        @test buffer.truncated[i] == x.trunc
    end
    @test count(buffer.terminated) == 3
    @test count(buffer.truncated) == 2
    @test all(a -> -1.0f0 <= a <= 1.0f0, buffer.actions[:, 1:length(buffer)])

    # Random warm-up actions follow the same storage path.
    _, success = Drill.collect_rollout!(buffer, cache, alg, env, 2; use_random_actions = true)
    @test success
    @test length(buffer) == (n_steps + 2) * n_envs
    stored = 1:length(buffer)
    @test buffer.next_observations[1, stored] == buffer.observations[1, stored] .+ 1
end

@testset "ReplayBuffer sampling and data loader" begin
    obs_space = Box(-1.0f0, 1.0f0, (3, 2))
    act_space = Box(-1.0f0, 1.0f0, (2,))
    buffer = ReplayBuffer(obs_space, act_space, 32)
    rng = Random.Xoshiro(7)
    @test_throws AssertionError Drill.sample_batch(buffer, 4, rng)

    n = 10
    obs = rand(rng, Float32, 3, 2, n)
    add_transitions!(
        buffer, obs, rand(rng, Float32, 2, n), Float32.(1:n),
        isodd.(1:n), falses(n), obs .+ 1.0f0,
    )

    sample = sample_batch(buffer, 7, rng)
    @test size(sample.observations) == (3, 2, 7)
    @test size(sample.next_observations) == (3, 2, 7)
    @test size(sample.actions) == (2, 7)
    @test size(sample.rewards) == (7,)
    @test size(sample.terminated) == (7,)
    @test size(sample.truncated) == (7,)
    @test eltype(sample.observations) == Float32
    @test eltype(sample.actions) == Float32
    @test eltype(sample.rewards) == Float32
    @test eltype(sample.terminated) == Bool
    # Only stored transitions are sampled, and the fields of one transition stay together.
    @test all(r -> r in 1:n, sample.rewards)
    for j in 1:7
        k = Int(sample.rewards[j])
        @test sample.observations[:, :, j] == obs[:, :, k]
        @test sample.next_observations[:, :, j] == obs[:, :, k] .+ 1.0f0
        @test sample.terminated[j] == isodd(k)
    end

    batch_size, batches = 4, 3
    loader = get_data_loader(buffer, batch_size, batches, true, false, rng)
    @test length(loader) == batches
    for b in loader
        @test size(b.observations) == (3, 2, batch_size)
        @test size(b.next_observations) == (3, 2, batch_size)
        @test size(b.actions) == (2, batch_size)
        @test size(b.rewards) == (batch_size,)
        @test size(b.terminated) == (batch_size,)
    end
end
