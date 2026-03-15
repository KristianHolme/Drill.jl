using Test
using Drill
using Random
include("setup.jl")
using .TestSetup

@testset "Environment interface validation" begin
    env = CustomEnv(8)

    @test hasmethod(Drill.observation_space, (typeof(env),))
    @test hasmethod(Drill.action_space, (typeof(env),))
    @test hasmethod(Drill.terminated, (typeof(env),))
    @test hasmethod(Drill.truncated, (typeof(env),))
    @test hasmethod(Drill.get_info, (typeof(env),))
    @test hasmethod(Drill.reset!, (typeof(env),))
    @test hasmethod(Drill.act!, (typeof(env), AbstractArray))

    @test hasmethod(Drill.observe, (typeof(env),))

    obs_space = Drill.observation_space(env)
    act_space = Drill.action_space(env)
    @test obs_space isa Box{Float32}
    @test act_space isa Box{Float32}
    @test obs_space.shape == (2,)
    @test act_space.shape == (2,)

    rng = Random.MersenneTwister(42)
    Random.seed!(env, rand(rng, UInt32))
    Drill.reset!(env)
    initial_obs = Drill.observe(env)
    @test length(initial_obs) == 2
    @test initial_obs ∈ obs_space
    @test !Drill.terminated(env)
    @test !Drill.truncated(env)

    action = rand(Float32, 2) .* 2.0f0 .- 1.0f0
    reward = Drill.act!(env, action)
    @test reward isa Float32
    @test reward ≥ 0.0f0

    next_obs = Drill.observe(env)
    term = Drill.terminated(env)
    trunc = Drill.truncated(env)
    info = Drill.get_info(env)
    @test length(next_obs) == 2

    obs = Drill.observe(env)
    @test length(obs) == 2
    @test obs ∈ obs_space
end

@testset "Environment episode completion" begin
    max_steps = 4
    env = CustomEnv(max_steps)

    Drill.reset!(env)
    action = rand(Float32, 2) .* 2.0f0 .- 1.0f0

    for step in 1:max_steps
        reward = Drill.act!(env, action)

        if step < max_steps
            @test reward ≈ 0.0f0
            @test !Drill.terminated(env)
            @test !Drill.truncated(env)
        else
            @test reward ≈ 1.0f0
            @test Drill.terminated(env)
            @test !Drill.truncated(env)
        end
    end
end

@testset "Infinite horizon environment validation" begin
    env = InfiniteHorizonEnv(4)

    @test Drill.observation_space(env) isa Box{Float32}
    @test Drill.action_space(env) isa Box{Float32}

    Drill.reset!(env)
    initial_obs = Drill.observe(env)
    @test length(initial_obs) == 1
    @test initial_obs[1] ≈ 0.0f0

    action = rand(Float32, 2) .* 2.0f0 .- 1.0f0
    for i in 1:20
        reward = Drill.act!(env, action)
        @test reward ≈ 1.0f0
        @test !Drill.terminated(env)
        @test !Drill.truncated(env)

        obs = Drill.observe(env)
        term = Drill.terminated(env)
        trunc = Drill.truncated(env)
        @test !term
        @test !trunc
        @test length(obs) == 1
    end
end

@testset "Environment wrapper validation" begin
    base_env = SimpleRewardEnv(6)
    constant_obs = [0.5f0, -0.3f0]
    wrapped_env = ConstantObsWrapper(base_env, constant_obs)

    @test Drill.observation_space(wrapped_env) == Drill.observation_space(base_env)
    @test Drill.action_space(wrapped_env) == Drill.action_space(base_env)

    Drill.reset!(wrapped_env)
    obs = Drill.observe(wrapped_env)
    @test obs == constant_obs
    @test !Drill.terminated(wrapped_env)
    @test !Drill.truncated(wrapped_env)

    action = rand(Float32, 2) .* 2.0f0 .- 1.0f0
    reward = Drill.act!(wrapped_env, action)
    @test reward isa Float32

    next_obs = Drill.observe(wrapped_env)
    term = Drill.terminated(wrapped_env)
    trunc = Drill.truncated(wrapped_env)
    info = Drill.get_info(wrapped_env)
    @test next_obs == constant_obs

    obs = Drill.observe(wrapped_env)
    @test obs == constant_obs
end

@testset "Environment space constraints" begin
    env = CustomEnv(8)
    obs_space = Drill.observation_space(env)
    act_space = Drill.action_space(env)

    rng = Random.MersenneTwister(123)
    for i in 1:10
        Random.seed!(env, rand(rng, UInt32))
        Drill.reset!(env)
        obs = Drill.observe(env)
        @test length(obs) == obs_space.shape[1]
        @test obs ∈ obs_space

        action = rand(Float32, act_space.shape...) .* 2.0f0 .- 1.0f0
        for step in 1:3
            Drill.act!(env, action)

            current_obs = Drill.observe(env)
            @test length(current_obs) == obs_space.shape[1]
            @test current_obs ∈ obs_space

            if Drill.terminated(env) || Drill.truncated(env)
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
    Random.seed!(env1, seed)
    Drill.reset!(env1)
    obs1 = Drill.observe(env1)
    push!(results1, copy(obs1))

    action = [0.5f0, -0.2f0]
    for i in 1:max_steps
        reward = Drill.act!(env1, action)
        obs = Drill.observe(env1)
        term = Drill.terminated(env1)
        trunc = Drill.truncated(env1)
        push!(results1, (copy(obs), reward, term, trunc))
        if term || trunc
            break
        end
    end

    results2 = []
    env2 = CustomEnv(max_steps)
    Random.seed!(env2, seed)
    Drill.reset!(env2)
    obs2 = Drill.observe(env2)
    push!(results2, copy(obs2))

    for i in 1:max_steps
        reward = Drill.act!(env2, action)
        obs = Drill.observe(env2)
        term = Drill.terminated(env2)
        trunc = Drill.truncated(env2)
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
