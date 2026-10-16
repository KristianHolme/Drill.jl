using Test
using Drill
using DrillInterface
using Random

@testset "reset! with seed: single and wrappers" begin
    mutable struct DummyEnv <: AbstractEnv
        rng::Random.AbstractRNG
    end

    DrillInterface.observation_space(::DummyEnv) = Box(Float32[0.0, 0.0], Float32[1.0, 1.0])
    DrillInterface.action_space(::DummyEnv) = Box(Float32[-1.0], Float32[1.0])
    DrillInterface.observe(env::DummyEnv) = rand(env.rng, Float32, 2)
    DrillInterface.terminated(::DummyEnv) = false
    DrillInterface.truncated(::DummyEnv) = false
    DrillInterface.act!(::DummyEnv, action) = 0.0f0
    DrillInterface.get_info(::DummyEnv) = Dict{String, Any}()
    function DrillInterface.reset!(env::DummyEnv; seed = nothing)
        isnothing(seed) || Random.seed!(env.rng, seed)
        return nothing
    end

    env = DummyEnv(Random.Xoshiro())
    @test isnothing(reset!(env; seed = 42))
    obs1 = observe(env)
    reset!(env; seed = 42)
    obs2 = observe(env)
    @test obs1 == obs2
    reset!(env; seed = 43)
    @test observe(env) != obs1

    base = DummyEnv(Random.Xoshiro())
    wrapped = ScalingWrapperEnv(base)
    @test isnothing(reset!(wrapped; seed = 123))
    w1 = observe(wrapped)
    reset!(wrapped; seed = 123)
    w2 = observe(wrapped)
    @test w1 == w2
end

@testset "reset! with seed: parallel envs" begin
    mutable struct DummyEnv2 <: AbstractEnv
        rng::Random.AbstractRNG
    end

    DrillInterface.observation_space(::DummyEnv2) = Box(Float32[0.0, 0.0], Float32[1.0, 1.0])
    DrillInterface.action_space(::DummyEnv2) = Box(Float32[-1.0], Float32[1.0])
    DrillInterface.observe(env::DummyEnv2) = rand(env.rng, Float32, 2)
    DrillInterface.terminated(::DummyEnv2) = false
    DrillInterface.truncated(::DummyEnv2) = false
    DrillInterface.act!(::DummyEnv2, action) = 0.0f0
    DrillInterface.get_info(::DummyEnv2) = Dict{String, Any}()
    function DrillInterface.reset!(env::DummyEnv2; seed = nothing)
        isnothing(seed) || Random.seed!(env.rng, seed)
        return nothing
    end

    envs_mt = [DummyEnv2(Random.Xoshiro()) for _ in 1:3]
    penv_mt = MultiThreadedParallelEnv(envs_mt)
    o1 = reset!(penv_mt; seed = 7)
    o2 = reset!(penv_mt; seed = 7)
    @test size(o1) == (2, 3)
    @test o1 == o2
    # Env i is seeded with seed + i - 1, so the envs differ from each other
    @test o1[:, 1] != o1[:, 2]

    envs_br = [DummyEnv2(Random.Xoshiro()) for _ in 1:2]
    penv_br = BroadcastedParallelEnv(envs_br)
    b1 = reset!(penv_br; seed = 99)
    b2 = reset!(penv_br; seed = 99)
    @test b1 == b2

    nenv = NormalizeWrapperEnv(penv_br; training = false)
    n1 = reset!(nenv; seed = 99)
    n2 = reset!(nenv; seed = 99)
    @test n1 ≈ n2

    envs1 = [DummyEnv2(Random.Xoshiro()) for _ in 1:2]
    envs2 = [DummyEnv2(Random.Xoshiro()) for _ in 1:3]
    p1 = BroadcastedParallelEnv(envs1)
    p2 = MultiThreadedParallelEnv(envs2)
    magent = MultiAgentParallelEnv([p1, p2])
    m1 = reset!(magent; seed = 2024)
    m2 = reset!(magent; seed = 2024)
    @test size(m1) == (2, 5)
    @test m1 == m2
end

@testset "reset! with seed on env without rng does not error" begin
    struct NoRNGEnv <: AbstractEnv end
    DrillInterface.observation_space(::NoRNGEnv) = Box(Float32[0.0], Float32[1.0])
    DrillInterface.action_space(::NoRNGEnv) = Box(Float32[-1.0], Float32[1.0])
    DrillInterface.observe(::NoRNGEnv) = Float32[rand()]
    DrillInterface.terminated(::NoRNGEnv) = false
    DrillInterface.truncated(::NoRNGEnv) = false
    DrillInterface.act!(::NoRNGEnv, action) = 0.0f0
    DrillInterface.get_info(::NoRNGEnv) = Dict{String, Any}()
    DrillInterface.reset!(::NoRNGEnv; seed = nothing) = nothing

    env = NoRNGEnv()
    @test isnothing(reset!(env; seed = 123))
end

@testset "reset! with seed reseeds env rng in place" begin
    mutable struct DummyEnv3 <: AbstractEnv
        rng::Random.AbstractRNG
    end

    DrillInterface.observation_space(::DummyEnv3) = Box(Float32[0.0, 0.0], Float32[1.0, 1.0])
    DrillInterface.action_space(::DummyEnv3) = Box(Float32[-1.0], Float32[1.0])
    DrillInterface.observe(env::DummyEnv3) = rand(env.rng, Float32, 2)
    DrillInterface.terminated(::DummyEnv3) = false
    DrillInterface.truncated(::DummyEnv3) = false
    DrillInterface.act!(::DummyEnv3, action) = 0.0f0
    DrillInterface.get_info(::DummyEnv3) = Dict{String, Any}()
    function DrillInterface.reset!(env::DummyEnv3; seed = nothing)
        isnothing(seed) || Random.seed!(env.rng, seed)
        return nothing
    end

    env = DummyEnv3(Random.Xoshiro())
    rng_before = env.rng
    reset!(env; seed = 11)
    a1 = observe(env)
    reset!(env; seed = 11)
    a2 = observe(env)
    @test a1 == a2
    @test env.rng === rng_before

    base = DummyEnv3(Random.Xoshiro())
    wrap = ScalingWrapperEnv(base)
    reset!(wrap; seed = 12)
    w1 = observe(wrap)
    reset!(wrap; seed = 12)
    w2 = observe(wrap)
    @test w1 == w2

    envs_b = [DummyEnv3(Random.Xoshiro()) for _ in 1:2]
    br = BroadcastedParallelEnv(envs_b)
    reset!(br; seed = 21)
    b1 = observe(br)
    reset!(br; seed = 21)
    b2 = observe(br)
    @test b1 == b2

    envs_mt = [DummyEnv3(Random.Xoshiro()) for _ in 1:3]
    mt = MultiThreadedParallelEnv(envs_mt)
    reset!(mt; seed = 33)
    m1 = observe(mt)
    reset!(mt; seed = 33)
    m2 = observe(mt)
    @test m1 == m2

    p1 = BroadcastedParallelEnv([DummyEnv3(Random.Xoshiro()) for _ in 1:2])
    p2 = MultiThreadedParallelEnv([DummyEnv3(Random.Xoshiro()) for _ in 1:1])
    ma = MultiAgentParallelEnv([p1, p2])
    reset!(ma; seed = 44)
    x1 = observe(ma)
    reset!(ma; seed = 44)
    x2 = observe(ma)
    @test x1 == x2
end

@testset "reset! with seed makes parallel rollouts reproducible" begin
    mutable struct TrackingEnvForSeeding <: AbstractEnv
        rng::Random.AbstractRNG
        value::Float32
        steps::Int
    end
    TrackingEnvForSeeding(rng) = TrackingEnvForSeeding(rng, 0.0f0, 0)
    DrillInterface.observation_space(::TrackingEnvForSeeding) = Box(Float32[0.0], Float32[1.0])
    DrillInterface.action_space(::TrackingEnvForSeeding) = Box(Float32[-1.0], Float32[1.0])
    DrillInterface.observe(env::TrackingEnvForSeeding) = Float32[env.value]
    DrillInterface.terminated(env::TrackingEnvForSeeding) = env.steps >= 4
    DrillInterface.truncated(::TrackingEnvForSeeding) = false
    function DrillInterface.act!(env::TrackingEnvForSeeding, action)
        reward = 1.0f0 - abs(action[1] - env.value)
        env.value = rand(env.rng, Float32)
        env.steps += 1
        return reward
    end
    function DrillInterface.reset!(env::TrackingEnvForSeeding; seed = nothing)
        isnothing(seed) || Random.seed!(env.rng, seed)
        env.value = rand(env.rng, Float32)
        env.steps = 0
        return nothing
    end

    function rollout(penv, seed)
        obs = [copy(reset!(penv; seed))]
        rewards = Vector{Float32}[]
        for _ in 1:10
            actions = [Float32[0.25f0] for _ in 1:number_of_envs(penv)]
            o, r, _, _, _, _ = step!(penv, actions)
            push!(obs, copy(o))
            push!(rewards, copy(r))
        end
        return obs, rewards
    end

    make_env() = BroadcastedParallelEnv([TrackingEnvForSeeding(Random.Xoshiro()) for _ in 1:3])

    obs1, rew1 = rollout(make_env(), 5)
    obs2, rew2 = rollout(make_env(), 5)
    obs3, _ = rollout(make_env(), 6)
    @test obs1 == obs2
    @test rew1 == rew2
    @test obs1 != obs3
    # Shifting the seed by one shifts the envs by one
    @test obs1[1][:, 2:3] == obs3[1][:, 1:2]
end
