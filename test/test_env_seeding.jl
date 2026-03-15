using Test
using Drill
using Random

@testset "Random.seed! single and wrappers" begin
    mutable struct DummyEnv <: AbstractEnv
        rng::Random.AbstractRNG
    end

    Drill.observation_space(::DummyEnv) = Box(Float32[0.0, 0.0], Float32[1.0, 1.0])
    Drill.action_space(::DummyEnv) = Box(Float32[-1.0], Float32[1.0])
    Drill.observe(env::DummyEnv) = rand(env.rng, Float32, 2)
    Drill.terminated(::DummyEnv) = false
    Drill.truncated(::DummyEnv) = false
    Drill.act!(::DummyEnv, action) = 0.0f0
    Drill.get_info(::DummyEnv) = Dict{String, Any}()
    Drill.reset!(::DummyEnv) = nothing

    env = DummyEnv(Random.Xoshiro())
    Random.seed!(env, 42)
    reset!(env)
    obs1 = observe(env)
    Random.seed!(env, 42)
    reset!(env)
    obs2 = observe(env)
    @test obs1 == obs2

    base = DummyEnv(Random.Xoshiro())
    wrapped = ScalingWrapperEnv(base)
    Random.seed!(wrapped, 123)
    reset!(wrapped)
    w1 = observe(wrapped)
    Random.seed!(wrapped, 123)
    reset!(wrapped)
    w2 = observe(wrapped)
    @test w1 == w2
end

@testset "Random.seed! parallel envs" begin
    mutable struct DummyEnv2 <: AbstractEnv
        rng::Random.AbstractRNG
    end

    Drill.observation_space(::DummyEnv2) = Box(Float32[0.0, 0.0], Float32[1.0, 1.0])
    Drill.action_space(::DummyEnv2) = Box(Float32[-1.0], Float32[1.0])
    Drill.observe(env::DummyEnv2) = rand(env.rng, Float32, 2)
    Drill.terminated(::DummyEnv2) = false
    Drill.truncated(::DummyEnv2) = false
    Drill.act!(::DummyEnv2, action) = 0.0f0
    Drill.get_info(::DummyEnv2) = Dict{String, Any}()
    Drill.reset!(::DummyEnv2) = nothing

    envs_mt = [DummyEnv2(Random.Xoshiro()) for _ in 1:3]
    penv_mt = MultiThreadedParallelEnv(envs_mt)
    Random.seed!(penv_mt, 7)
    reset!(penv_mt)
    o1 = observe(penv_mt)
    Random.seed!(penv_mt, 7)
    reset!(penv_mt)
    o2 = observe(penv_mt)
    @test all([o1[i] == o2[i] for i in eachindex(o1)])

    envs_br = [DummyEnv2(Random.Xoshiro()) for _ in 1:2]
    penv_br = BroadcastedParallelEnv(envs_br)
    Random.seed!(penv_br, 99)
    reset!(penv_br)
    b1 = observe(penv_br)
    Random.seed!(penv_br, 99)
    reset!(penv_br)
    b2 = observe(penv_br)
    @test all([b1[i] == b2[i] for i in eachindex(b1)])

    nenv = NormalizeWrapperEnv(penv_br; training = false)
    Random.seed!(nenv, 99)
    reset!(nenv)
    n1 = observe(nenv)
    Random.seed!(nenv, 99)
    reset!(nenv)
    n2 = observe(nenv)
    @test all([n1[i] == n2[i] for i in eachindex(n1)])

    envs1 = [DummyEnv2(Random.Xoshiro()) for _ in 1:2]
    envs2 = [DummyEnv2(Random.Xoshiro()) for _ in 1:3]
    p1 = BroadcastedParallelEnv(envs1)
    p2 = MultiThreadedParallelEnv(envs2)
    magent = MultiAgentParallelEnv([p1, p2])
    Random.seed!(magent, 2024)
    reset!(magent)
    m1 = observe(magent)
    Random.seed!(magent, 2024)
    reset!(magent)
    m2 = observe(magent)
    @test length(m1) == length(m2)
    @test all([m1[i] == m2[i] for i in eachindex(m1)])
end

@testset "Random.seed! no-rng env does not error" begin
    struct NoRNGEnv <: AbstractEnv end
    Drill.observation_space(::NoRNGEnv) = Box(Float32[0.0], Float32[1.0])
    Drill.action_space(::NoRNGEnv) = Box(Float32[-1.0], Float32[1.0])
    Drill.observe(::NoRNGEnv) = Float32[rand()]
    Drill.terminated(::NoRNGEnv) = false
    Drill.truncated(::NoRNGEnv) = false
    Drill.act!(::NoRNGEnv, action) = 0.0f0
    Drill.get_info(::NoRNGEnv) = Dict{String, Any}()
    Drill.reset!(::NoRNGEnv) = nothing

    env = NoRNGEnv()
    Random.seed!(env, 123)
    @test true
end

@testset "Random.seed! mutates in-place" begin
    mutable struct DummyEnv3 <: AbstractEnv
        rng::Random.AbstractRNG
    end

    Drill.observation_space(::DummyEnv3) = Box(Float32[0.0, 0.0], Float32[1.0, 1.0])
    Drill.action_space(::DummyEnv3) = Box(Float32[-1.0], Float32[1.0])
    Drill.observe(env::DummyEnv3) = rand(env.rng, Float32, 2)
    Drill.terminated(::DummyEnv3) = false
    Drill.truncated(::DummyEnv3) = false
    Drill.act!(::DummyEnv3, action) = 0.0f0
    Drill.get_info(::DummyEnv3) = Dict{String, Any}()
    Drill.reset!(::DummyEnv3) = nothing

    env = DummyEnv3(Random.Xoshiro())
    Random.seed!(env, 11)
    reset!(env)
    a1 = observe(env)
    Random.seed!(env, 11)
    reset!(env)
    a2 = observe(env)
    @test a1 == a2

    base = DummyEnv3(Random.Xoshiro())
    wrap = ScalingWrapperEnv(base)
    Random.seed!(wrap, 12)
    reset!(wrap)
    w1 = observe(wrap)
    Random.seed!(wrap, 12)
    reset!(wrap)
    w2 = observe(wrap)
    @test w1 == w2

    envs_b = [DummyEnv3(Random.Xoshiro()) for _ in 1:2]
    br = BroadcastedParallelEnv(envs_b)
    Random.seed!(br, 21)
    reset!(br)
    b1 = observe(br)
    Random.seed!(br, 21)
    reset!(br)
    b2 = observe(br)
    @test all([b1[i] == b2[i] for i in eachindex(b1)])

    envs_mt = [DummyEnv3(Random.Xoshiro()) for _ in 1:3]
    mt = MultiThreadedParallelEnv(envs_mt)
    Random.seed!(mt, 33)
    reset!(mt)
    m1 = observe(mt)
    Random.seed!(mt, 33)
    reset!(mt)
    m2 = observe(mt)
    @test all([m1[i] == m2[i] for i in eachindex(m1)])

    p1 = BroadcastedParallelEnv([DummyEnv3(Random.Xoshiro()) for _ in 1:2])
    p2 = MultiThreadedParallelEnv([DummyEnv3(Random.Xoshiro()) for _ in 1:1])
    ma = MultiAgentParallelEnv([p1, p2])
    Random.seed!(ma, 44)
    reset!(ma)
    x1 = observe(ma)
    Random.seed!(ma, 44)
    reset!(ma)
    x2 = observe(ma)
    @test all([x1[i] == x2[i] for i in eachindex(x1)])
end
