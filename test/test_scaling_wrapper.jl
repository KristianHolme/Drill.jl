using Test
using Drill
using Random

@testset "ScalingWrapperEnv basic construction" begin
    struct TestScalingEnv <: AbstractEnv
        rng::Random.AbstractRNG
    end
    TestScalingEnv() = TestScalingEnv(Random.Xoshiro(42))

    DrillInterface.observation_space(::TestScalingEnv) = Box(Float32[10.0, 20.0], Float32[50.0, 80.0])
    DrillInterface.action_space(::TestScalingEnv) = Box(Float32[2.0, -5.0], Float32[8.0, 15.0])
    DrillInterface.observe(env::TestScalingEnv) = Float32[30.0, 50.0]
    DrillInterface.terminated(::TestScalingEnv) = false
    DrillInterface.truncated(::TestScalingEnv) = false
    DrillInterface.act!(::TestScalingEnv, action) = 1.0f0
    DrillInterface.get_info(::TestScalingEnv) = Dict("test" => "info")
    DrillInterface.reset!(::TestScalingEnv) = nothing

    base_env = TestScalingEnv()
    scaled_env = ScalingWrapperEnv(base_env)

    @test scaled_env isa ScalingWrapperEnv
    @test unwrap(scaled_env) === base_env

    scaled_obs_space = observation_space(scaled_env)
    scaled_act_space = action_space(scaled_env)

    @test all(scaled_obs_space.low .== -1.0f0)
    @test all(scaled_obs_space.high .== 1.0f0)
    @test all(scaled_act_space.low .== -1.0f0)
    @test all(scaled_act_space.high .== 1.0f0)

    @test isequal(scaled_env.orig_observation_space, observation_space(base_env))
    @test isequal(scaled_env.orig_action_space, action_space(base_env))
end

@testset "ScalingWrapperEnv observation scaling" begin
    struct ObsTestEnv <: AbstractEnv
        obs_value::Vector{Float32}
    end

    DrillInterface.observation_space(::ObsTestEnv) = Box(Float32[0.0, -10.0, 5.0], Float32[10.0, 10.0, 25.0])
    DrillInterface.action_space(::ObsTestEnv) = Box(Float32[-1.0], Float32[1.0])
    DrillInterface.observe(env::ObsTestEnv) = env.obs_value
    DrillInterface.terminated(::ObsTestEnv) = false
    DrillInterface.truncated(::ObsTestEnv) = false
    DrillInterface.act!(::ObsTestEnv, action) = 0.0f0
    DrillInterface.get_info(::ObsTestEnv) = Dict()
    DrillInterface.reset!(::ObsTestEnv) = nothing

    base_env = ObsTestEnv(Float32[5.0, 0.0, 15.0])
    scaled_env = ScalingWrapperEnv(base_env)

    scaled_obs = observe(scaled_env)

    expected = Float32[0.0, 0.0, 0.0]
    @test all(abs.(scaled_obs .- expected) .< 1.0e-6)

    base_env.obs_value .= Float32[0.0, -10.0, 5.0]
    scaled_obs = observe(scaled_env)
    @test all(abs.(scaled_obs .- Float32[-1.0, -1.0, -1.0]) .< 1.0e-6)

    base_env.obs_value .= Float32[10.0, 10.0, 25.0]
    scaled_obs = observe(scaled_env)
    @test all(abs.(scaled_obs .- Float32[1.0, 1.0, 1.0]) .< 1.0e-6)
end

@testset "ScalingWrapperEnv action scaling" begin
    mutable struct ActionTestEnv <: AbstractEnv
        last_action::Vector{Float32}
    end
    ActionTestEnv() = ActionTestEnv(Float32[])

    DrillInterface.observation_space(::ActionTestEnv) = Box(Float32[-1.0], Float32[1.0])
    DrillInterface.action_space(::ActionTestEnv) = Box(Float32[2.0, -5.0, 0.0], Float32[8.0, 15.0, 10.0])
    DrillInterface.observe(::ActionTestEnv) = Float32[0.0]
    DrillInterface.terminated(::ActionTestEnv) = false
    DrillInterface.truncated(::ActionTestEnv) = false
    function DrillInterface.act!(env::ActionTestEnv, action)
        env.last_action = copy(action)
        return 1.0f0
    end
    DrillInterface.get_info(::ActionTestEnv) = Dict()
    DrillInterface.reset!(::ActionTestEnv) = nothing

    base_env = ActionTestEnv()
    scaled_env = ScalingWrapperEnv(base_env)

    scaled_action = Float32[0.0, 0.0, 0.0]
    act!(scaled_env, scaled_action)

    expected = Float32[5.0, 5.0, 5.0]
    @test all(abs.(base_env.last_action .- expected) .< 1.0e-6)

    scaled_action = Float32[-1.0, -1.0, -1.0]
    act!(scaled_env, scaled_action)
    expected = Float32[2.0, -5.0, 0.0]
    @test all(abs.(base_env.last_action .- expected) .< 1.0e-6)

    scaled_action = Float32[1.0, 1.0, 1.0]
    act!(scaled_env, scaled_action)
    expected = Float32[8.0, 15.0, 10.0]
    @test all(abs.(base_env.last_action .- expected) .< 1.0e-6)
end

@testset "ScalingWrapperEnv method forwarding" begin
    mutable struct ForwardingTestEnv <: AbstractEnv
        _terminated::Bool
        _truncated::Bool
        _info::Dict{String, Any}
        reset_called::Bool
    end
    ForwardingTestEnv() = ForwardingTestEnv(false, false, Dict("key" => "value"), false)

    DrillInterface.observation_space(::ForwardingTestEnv) = Box(Float32[0.0], Float32[1.0])
    DrillInterface.action_space(::ForwardingTestEnv) = Box(Float32[0.0], Float32[1.0])
    DrillInterface.observe(::ForwardingTestEnv) = Float32[0.5]
    DrillInterface.terminated(env::ForwardingTestEnv) = env._terminated
    DrillInterface.truncated(env::ForwardingTestEnv) = env._truncated
    DrillInterface.act!(::ForwardingTestEnv, action) = 1.0f0
    DrillInterface.get_info(env::ForwardingTestEnv) = env._info
    function DrillInterface.reset!(env::ForwardingTestEnv)
        env.reset_called = true
        nothing
    end

    base_env = ForwardingTestEnv()
    scaled_env = ScalingWrapperEnv(base_env)

    @test !base_env.reset_called
    reset!(scaled_env)
    @test base_env.reset_called

    @test !terminated(scaled_env)
    base_env._terminated = true
    @test terminated(scaled_env)

    @test !truncated(scaled_env)
    base_env._truncated = true
    @test truncated(scaled_env)

    info = get_info(scaled_env)
    @test info["key"] == "value"
end

@testset "ScalingWrapperEnv edge cases" begin
    struct EdgeCaseEnv <: AbstractEnv end

    DrillInterface.observation_space(::EdgeCaseEnv) = Box(Float32[0.0, -1.0], Float32[0.0, -1.0])
    DrillInterface.action_space(::EdgeCaseEnv) = Box(Float32[5.0], Float32[5.0])
    DrillInterface.observe(::EdgeCaseEnv) = Float32[0.0, -1.0]
    DrillInterface.terminated(::EdgeCaseEnv) = false
    DrillInterface.truncated(::EdgeCaseEnv) = false
    DrillInterface.act!(::EdgeCaseEnv, action) = 0.0f0
    DrillInterface.get_info(::EdgeCaseEnv) = Dict()
    DrillInterface.reset!(::EdgeCaseEnv) = nothing

    base_env = EdgeCaseEnv()
    scaled_env = ScalingWrapperEnv(base_env)

    scaled_obs = observe(scaled_env)
    @test length(scaled_obs) == 2

    scaled_action = Float32[0.0]
    reward = act!(scaled_env, scaled_action)
    @test reward == 0.0f0
end

@testset "ScalingWrapperEnv large ranges" begin
    struct LargeRangeEnv <: AbstractEnv end

    DrillInterface.observation_space(::LargeRangeEnv) = Box(Float32[-1000.0, -500.0], Float32[2000.0, 1500.0])
    DrillInterface.action_space(::LargeRangeEnv) = Box(Float32[-100.0], Float32[300.0])
    DrillInterface.observe(::LargeRangeEnv) = Float32[500.0, 500.0]
    DrillInterface.terminated(::LargeRangeEnv) = false
    DrillInterface.truncated(::LargeRangeEnv) = false
    DrillInterface.act!(::LargeRangeEnv, action) = action[1]
    DrillInterface.get_info(::LargeRangeEnv) = Dict()
    DrillInterface.reset!(::LargeRangeEnv) = nothing

    base_env = LargeRangeEnv()
    scaled_env = ScalingWrapperEnv(base_env)

    scaled_obs = observe(scaled_env)
    expected = Float32[0.0, 0.0]
    @test all(abs.(scaled_obs .- expected) .< 1.0e-5)

    scaled_action = Float32[0.5]
    reward = act!(scaled_env, scaled_action)
    @test abs(reward - 200.0f0) < 1.0e-5
end

@testset "ScalingWrapperEnv multi-dimensional spaces" begin
    mutable struct MultiDimEnv <: AbstractEnv
        last_action::Array{Float32}
    end
    MultiDimEnv() = MultiDimEnv(Float32[])

    DrillInterface.observation_space(::MultiDimEnv) = Box(Float32[0.0 5.0; 10.0 15.0; 20.0 25.0], Float32[4.0 9.0; 14.0 19.0; 24.0 29.0])
    DrillInterface.action_space(::MultiDimEnv) = Box(Float32[1.0 3.0; 2.0 4.0], Float32[2.0 6.0; 5.0 8.0])
    DrillInterface.observe(::MultiDimEnv) = Float32[2.0 7.0; 12.0 17.0; 22.0 27.0]
    DrillInterface.terminated(::MultiDimEnv) = false
    DrillInterface.truncated(::MultiDimEnv) = false
    function DrillInterface.act!(env::MultiDimEnv, action)
        env.last_action = copy(action)
        return 1.0f0
    end
    DrillInterface.get_info(::MultiDimEnv) = Dict()
    DrillInterface.reset!(::MultiDimEnv) = nothing

    base_env = MultiDimEnv()
    scaled_env = ScalingWrapperEnv(base_env)

    scaled_obs = observe(scaled_env)
    @test size(scaled_obs) == (3, 2)
    @test all(abs.(scaled_obs) .< 1.0e-5)

    scaled_action = Float32[0.0 -0.5; 0.5 1.0]
    act!(scaled_env, scaled_action)

    @test size(base_env.last_action) == (2, 2)
    expected = Float32[1.5 3.75; 4.25 8.0]
    @test all(abs.(base_env.last_action .- expected) .< 1.0e-5)
end

@testset "ScalingWrapperEnv Random seeding" begin
    mutable struct SeededTestEnv <: AbstractEnv
        rng::Random.AbstractRNG
        obs_counter::Int
    end
    SeededTestEnv() = SeededTestEnv(Random.Xoshiro(1234), 0)

    DrillInterface.observation_space(::SeededTestEnv) = Box(Float32[0.0], Float32[10.0])
    DrillInterface.action_space(::SeededTestEnv) = Box(Float32[0.0], Float32[1.0])
    function DrillInterface.observe(env::SeededTestEnv)
        env.obs_counter += 1
        return Float32[rand(env.rng) * 10.0]
    end
    DrillInterface.terminated(::SeededTestEnv) = false
    DrillInterface.truncated(::SeededTestEnv) = false
    DrillInterface.act!(::SeededTestEnv, action) = 0.0f0
    DrillInterface.get_info(::SeededTestEnv) = Dict()
    DrillInterface.reset!(::SeededTestEnv) = nothing

    base_env = SeededTestEnv()
    scaled_env = ScalingWrapperEnv(base_env)

    obs1 = observe(scaled_env)
    obs2 = observe(scaled_env)
    @test obs1 != obs2

    Random.seed!(scaled_env, 42)
    obs3 = observe(scaled_env)
    obs4 = observe(scaled_env)

    Random.seed!(scaled_env, 42)
    base_env.obs_counter = 2
    obs5 = observe(scaled_env)
    obs6 = observe(scaled_env)

    @test obs3 == obs5
    @test obs4 == obs6
end

@testset "ScalingWrapperEnv integration test" begin
    mutable struct IntegrationTestEnv <: AbstractEnv
        state::Float32
        step_count::Int
        max_steps::Int
        rng::Random.AbstractRNG
    end
    IntegrationTestEnv() = IntegrationTestEnv(0.0f0, 0, 10, Random.Xoshiro(123))

    DrillInterface.observation_space(::IntegrationTestEnv) = Box(Float32[-5.0], Float32[15.0])
    DrillInterface.action_space(::IntegrationTestEnv) = Box(Float32[-2.0], Float32[2.0])
    DrillInterface.observe(env::IntegrationTestEnv) = Float32[env.state]
    DrillInterface.terminated(env::IntegrationTestEnv) = env.step_count >= env.max_steps
    DrillInterface.truncated(::IntegrationTestEnv) = false
    function DrillInterface.act!(env::IntegrationTestEnv, action)
        env.state = clamp(env.state + action[1], -5.0f0, 15.0f0)
        env.step_count += 1
        return Float32(10.0 - abs(env.state - 5.0))
    end
    DrillInterface.get_info(env::IntegrationTestEnv) = Dict("step" => env.step_count, "state" => env.state)
    function DrillInterface.reset!(env::IntegrationTestEnv)
        env.state = 0.0f0
        env.step_count = 0
        nothing
    end

    base_env = IntegrationTestEnv()
    scaled_env = ScalingWrapperEnv(base_env)

    reset!(scaled_env)

    let tot_reward = 0.0f0, episode_length = 0
        while !terminated(scaled_env) && episode_length < 20
            obs = observe(scaled_env)
            @test all(obs .>= -1.0f0) && all(obs .<= 1.0f0)

            action = Float32[0.1]
            reward = act!(scaled_env, action)
            tot_reward += reward
            episode_length += 1

            info = get_info(scaled_env)
            @test haskey(info, "step")
            @test haskey(info, "state")
        end

        @test episode_length == base_env.max_steps
        @test terminated(scaled_env)
        @test !truncated(scaled_env)
        @test tot_reward > 0.0f0
    end
end
