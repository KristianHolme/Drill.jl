using Test
using Drill
using DrillInterface
using Random
include("setup.jl")
using .TestSetup

@testset "Environment interface validation" begin
    env = CustomEnv(8)

    @test hasmethod(DrillInterface.observation_space, (typeof(env),))
    @test hasmethod(DrillInterface.action_space, (typeof(env),))
    @test hasmethod(DrillInterface.terminated, (typeof(env),))
    @test hasmethod(DrillInterface.truncated, (typeof(env),))
    @test hasmethod(DrillInterface.get_info, (typeof(env),))
    @test hasmethod(DrillInterface.reset!, (typeof(env),))
    @test hasmethod(DrillInterface.act!, (typeof(env), AbstractArray))

    @test hasmethod(DrillInterface.observe, (typeof(env),))

    obs_space = DrillInterface.observation_space(env)
    act_space = DrillInterface.action_space(env)
    @test obs_space isa Box{Float32}
    @test act_space isa Box{Float32}
    @test obs_space.shape == (2,)
    @test act_space.shape == (2,)

    rng = Random.MersenneTwister(42)
    @test isnothing(DrillInterface.reset!(env; seed = rand(rng, UInt32)))
    initial_obs = DrillInterface.observe(env)
    @test length(initial_obs) == 2
    @test initial_obs ∈ obs_space
    @test !DrillInterface.terminated(env)
    @test !DrillInterface.truncated(env)

    action = rand(Float32, 2) .* 2.0f0 .- 1.0f0
    reward = DrillInterface.act!(env, action)
    @test reward isa Float32
    @test reward ≥ 0.0f0

    next_obs = DrillInterface.observe(env)
    term = DrillInterface.terminated(env)
    trunc = DrillInterface.truncated(env)
    info = DrillInterface.get_info(env)
    @test length(next_obs) == 2

    obs = DrillInterface.observe(env)
    @test length(obs) == 2
    @test obs ∈ obs_space
end

@testset "Environment episode completion" begin
    max_steps = 4
    env = CustomEnv(max_steps)

    DrillInterface.reset!(env)
    action = rand(Float32, 2) .* 2.0f0 .- 1.0f0

    for step in 1:max_steps
        reward = DrillInterface.act!(env, action)

        if step < max_steps
            @test reward ≈ 0.0f0
            @test !DrillInterface.terminated(env)
            @test !DrillInterface.truncated(env)
        else
            @test reward ≈ 1.0f0
            @test DrillInterface.terminated(env)
            @test !DrillInterface.truncated(env)
        end
    end
end

@testset "Infinite horizon environment validation" begin
    env = InfiniteHorizonEnv(4)

    @test DrillInterface.observation_space(env) isa Box{Float32}
    @test DrillInterface.action_space(env) isa Box{Float32}

    DrillInterface.reset!(env)
    initial_obs = DrillInterface.observe(env)
    @test length(initial_obs) == 1
    @test initial_obs[1] ≈ 0.0f0

    action = rand(Float32, 2) .* 2.0f0 .- 1.0f0
    for i in 1:20
        reward = DrillInterface.act!(env, action)
        @test reward ≈ 1.0f0
        @test !DrillInterface.terminated(env)
        @test !DrillInterface.truncated(env)

        obs = DrillInterface.observe(env)
        term = DrillInterface.terminated(env)
        trunc = DrillInterface.truncated(env)
        @test !term
        @test !trunc
        @test length(obs) == 1
    end
end

@testset "Environment wrapper validation" begin
    base_env = SimpleRewardEnv(6)
    constant_obs = [0.5f0, -0.3f0]
    wrapped_env = ConstantObsWrapper(base_env, constant_obs)

    @test DrillInterface.observation_space(wrapped_env) == DrillInterface.observation_space(base_env)
    @test DrillInterface.action_space(wrapped_env) == DrillInterface.action_space(base_env)

    DrillInterface.reset!(wrapped_env)
    obs = DrillInterface.observe(wrapped_env)
    @test obs == constant_obs
    @test !DrillInterface.terminated(wrapped_env)
    @test !DrillInterface.truncated(wrapped_env)

    action = rand(Float32, 2) .* 2.0f0 .- 1.0f0
    reward = DrillInterface.act!(wrapped_env, action)
    @test reward isa Float32

    next_obs = DrillInterface.observe(wrapped_env)
    term = DrillInterface.terminated(wrapped_env)
    trunc = DrillInterface.truncated(wrapped_env)
    info = DrillInterface.get_info(wrapped_env)
    @test next_obs == constant_obs

    obs = DrillInterface.observe(wrapped_env)
    @test obs == constant_obs
end

@testset "Environment space constraints" begin
    env = CustomEnv(8)
    obs_space = DrillInterface.observation_space(env)
    act_space = DrillInterface.action_space(env)

    rng = Random.MersenneTwister(123)
    for i in 1:10
        DrillInterface.reset!(env; seed = rand(rng, UInt32))
        obs = DrillInterface.observe(env)
        @test length(obs) == obs_space.shape[1]
        @test obs ∈ obs_space

        action = rand(Float32, act_space.shape...) .* 2.0f0 .- 1.0f0
        for step in 1:3
            DrillInterface.act!(env, action)

            current_obs = DrillInterface.observe(env)
            @test length(current_obs) == obs_space.shape[1]
            @test current_obs ∈ obs_space

            if DrillInterface.terminated(env) || DrillInterface.truncated(env)
                break
            end
        end
    end
end

@testset "Environment reproducibility" begin
    seed = 42
    max_steps = 6

    results1 = []
    env1 = CustomEnv(max_steps)
    DrillInterface.reset!(env1; seed)
    obs1 = DrillInterface.observe(env1)
    push!(results1, copy(obs1))

    action = [0.5f0, -0.2f0]
    for i in 1:max_steps
        reward = DrillInterface.act!(env1, action)
        obs = DrillInterface.observe(env1)
        term = DrillInterface.terminated(env1)
        trunc = DrillInterface.truncated(env1)
        push!(results1, (copy(obs), reward, term, trunc))
        if term || trunc
            break
        end
    end

    results2 = []
    env2 = CustomEnv(max_steps)
    DrillInterface.reset!(env2; seed)
    obs2 = DrillInterface.observe(env2)
    push!(results2, copy(obs2))

    for i in 1:max_steps
        reward = DrillInterface.act!(env2, action)
        obs = DrillInterface.observe(env2)
        term = DrillInterface.terminated(env2)
        trunc = DrillInterface.truncated(env2)
        push!(results2, (copy(obs), reward, term, trunc))
        if term || trunc
            break
        end
    end

    @test length(results1) == length(results2)
    @test results1[1] ≈ results2[1]

    @test all(
        i -> begin
            obs1, reward1, term1, trunc1 = results1[i]
            obs2, reward2, term2, trunc2 = results2[i]
            obs1 ≈ obs2 && reward1 ≈ reward2 && term1 == term2 && trunc1 == trunc2
        end, eachindex(results1)[2:end]
    )
end

# Deterministic env without get_info: observes its step count, reward 1 per step, and
# ends after `max_steps` steps by termination or (if `truncate`) truncation.
mutable struct CounterEnv <: AbstractEnv
    max_steps::Int
    truncate::Bool
    steps::Int
end
CounterEnv(max_steps::Int; truncate::Bool = false) = CounterEnv(max_steps, truncate, 0)

DrillInterface.observation_space(::CounterEnv) = Box(0.0f0, 100.0f0, (1,))
DrillInterface.action_space(::CounterEnv) = Box(-1.0f0, 1.0f0, (1,))
DrillInterface.observe(env::CounterEnv) = Float32[env.steps]
DrillInterface.terminated(env::CounterEnv) = !env.truncate && env.steps >= env.max_steps
DrillInterface.truncated(env::CounterEnv) = env.truncate && env.steps >= env.max_steps
function DrillInterface.act!(env::CounterEnv, action)
    env.steps += 1
    return 1.0f0
end
function DrillInterface.reset!(env::CounterEnv; seed = nothing)
    env.steps = 0
    return nothing
end

@testset "check_env on single and parallel envs" begin
    @test check_env(CustomEnv(8); verbose = false)
    @test check_env(CounterEnv(3); verbose = false)
    @test check_env(BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2]); verbose = false)
    @test check_env(MultiThreadedParallelEnv([CustomEnv(8) for _ in 1:3]); verbose = false)
    @test check_env(MonitorWrapperEnv(BroadcastedParallelEnv([CounterEnv(3) for _ in 1:2])); verbose = false)
end

@testset "Parallel env step! return shapes" begin
    make_envs = Dict(
        "BroadcastedParallelEnv" => () -> BroadcastedParallelEnv([CustomEnv(8) for _ in 1:3]),
        "MultiThreadedParallelEnv" => () -> MultiThreadedParallelEnv([CustomEnv(8) for _ in 1:3]),
        "MultiAgentParallelEnv" => () -> MultiAgentParallelEnv(
            [
                BroadcastedParallelEnv([CustomEnv(8) for _ in 1:1]),
                BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2]),
            ]
        ),
    )
    for (name, make_env) in make_envs
        @testset "$name" begin
            penv = make_env()
            n = number_of_envs(penv)
            @test n == 3
            @test !(penv isa AbstractEnv)
            @test penv isa AbstractParallelEnv

            obs = reset!(penv; seed = 1)
            @test size(obs) == (2, n)
            @test eltype(obs) == Float32
            @test size(observe(penv)) == (2, n)

            actions = rand(action_space(penv), n)
            result = step!(penv, actions)
            @test result isa Tuple
            @test length(result) == 6
            obs, rewards, terms, truncs, final_obs, infos = result
            @test obs isa Matrix{Float32}
            @test size(obs) == (2, n)
            @test rewards isa Vector{Float32}
            @test length(rewards) == n
            @test terms isa AbstractVector{Bool}
            @test length(terms) == n
            @test truncs isa AbstractVector{Bool}
            @test length(truncs) == n
            @test size(final_obs) == (2, n)
            @test eltype(final_obs) == Float32
            @test infos isa AbstractVector
            @test length(infos) == n
            @test all(info -> info isa Dict, infos)
            for i in 1:n
                @test collect(DrillInterface.observation_slot(obs, i)) == obs[:, i]
                @test obs[:, i] ∈ observation_space(penv)
            end
            @test_throws AssertionError step!(penv, actions[1:(n - 1)])
        end
    end
end

@testset "Parallel env infos are nothing when sub-env infos are nothing" begin
    penv = BroadcastedParallelEnv([CounterEnv(3) for _ in 1:2])
    reset!(penv)
    _, _, _, _, _, infos = step!(penv, [Float32[0.0] for _ in 1:2])
    @test isnothing(infos)

    multi = MultiAgentParallelEnv(
        [
            BroadcastedParallelEnv([CounterEnv(3) for _ in 1:2]),
            MultiThreadedParallelEnv([CounterEnv(3) for _ in 1:1]),
        ]
    )
    reset!(multi)
    _, _, _, _, _, infos = step!(multi, [Float32[0.0] for _ in 1:3])
    @test isnothing(infos)
end

@testset "Parallel env final_obs holds the pre-reset observation" begin
    for (PEnv, truncate) in [
            (BroadcastedParallelEnv, false), (BroadcastedParallelEnv, true),
            (MultiThreadedParallelEnv, false), (MultiThreadedParallelEnv, true),
        ]
        # Env 1 ends after 2 steps, env 2 after 3 steps
        penv = PEnv([CounterEnv(2; truncate), CounterEnv(3; truncate)])
        obs = reset!(penv)
        @test obs == Float32[0 0]
        actions = [Float32[0.0] for _ in 1:2]

        obs, rewards, terms, truncs, final_obs, _ = step!(penv, actions)
        @test obs == Float32[1 1]
        @test !any(terms) && !any(truncs)

        obs, rewards, terms, truncs, final_obs, _ = step!(penv, actions)
        done = truncate ? truncs : terms
        @test done == [true, false]
        @test !any(truncate ? terms : truncs)
        @test final_obs[:, 1] == Float32[2]
        @test obs[:, 1] == Float32[0]
        @test obs[:, 2] == Float32[2]

        obs, rewards, terms, truncs, final_obs, _ = step!(penv, actions)
        done = truncate ? truncs : terms
        @test done == [false, true]
        @test final_obs[:, 2] == Float32[3]
        @test obs == Float32[1 0]
    end

    # MultiAgentParallelEnv keeps final_obs aligned with the stacked env order
    multi = MultiAgentParallelEnv(
        [
            BroadcastedParallelEnv([CounterEnv(3), CounterEnv(3)]),
            BroadcastedParallelEnv([CounterEnv(1)]),
        ]
    )
    reset!(multi)
    obs, _, terms, _, final_obs, _ = step!(multi, [Float32[0.0] for _ in 1:3])
    @test terms == [false, false, true]
    @test final_obs[:, 3] == Float32[1]
    @test obs == Float32[1 1 0]

    # Same check with an env whose observations are random but read without side effects
    seed = 17
    max_steps = 4
    ref = TrackingTargetEnv(max_steps, Random.Xoshiro())
    reset!(ref; seed)
    for _ in 1:max_steps
        act!(ref, Float32[0.0])
    end
    expected_final = observe(ref)
    reset!(ref)
    expected_next = observe(ref)

    penv = BroadcastedParallelEnv([TrackingTargetEnv(max_steps, Random.Xoshiro()) for _ in 1:2])
    reset!(penv; seed)
    for _ in 1:max_steps
        obs, _, terms, _, final_obs, _ = step!(penv, [Float32[0.0] for _ in 1:2])
    end
    @test all(terms)
    @test final_obs[:, 1] == expected_final
    @test obs[:, 1] == expected_next
    @test final_obs[:, 1] != obs[:, 1]
end

@testset "MonitorWrapperEnv tracks last finished episodes" begin
    penv = BroadcastedParallelEnv([CounterEnv(2), CounterEnv(3)])
    monitor = MonitorWrapperEnv(penv)
    @test number_of_envs(monitor) == 2
    obs = reset!(monitor)
    @test obs == Float32[0 0]
    @test monitor.last_episode_lengths == [0, 0]
    @test monitor.last_episode_returns == Float32[0, 0]

    actions = [Float32[0.0] for _ in 1:2]
    step!(monitor, actions)
    @test monitor.last_episode_lengths == [0, 0]
    _, _, terms, _, _, infos = step!(monitor, actions)
    @test terms == [true, false]
    @test monitor.last_episode_lengths == [2, 0]
    @test monitor.last_episode_returns == Float32[2, 0]
    @test isnothing(infos)

    for _ in 1:4
        step!(monitor, actions)
    end
    # After 6 steps: env 1 finished 3 episodes, env 2 finished 2 episodes
    @test monitor.last_episode_lengths == [2, 3]
    @test monitor.last_episode_returns == Float32[2, 3]
    @test length(monitor.episode_stats.episode_lengths) == 5
    @test sort(collect(monitor.episode_stats.episode_lengths)) == [2, 2, 2, 3, 3]

    # Infos from sub-envs pass through unchanged, without an "episode" entry
    monitor_info = MonitorWrapperEnv(BroadcastedParallelEnv([CustomEnv(2) for _ in 1:2]))
    reset!(monitor_info)
    infos = nothing
    for _ in 1:2
        _, _, _, _, _, infos = step!(monitor_info, rand(action_space(monitor_info), 2))
    end
    @test monitor_info.last_episode_lengths == [2, 2]
    @test monitor_info.last_episode_returns == Float32[1, 1]
    @test all(info -> !haskey(info, "episode"), infos)

    # Episodes cut short by reset! are not recorded
    step!(monitor, actions)
    reset!(monitor)
    @test monitor.current_episode_lengths == [0, 0]
    @test length(monitor.episode_stats.episode_lengths) == 5
end
