using TestItems

@testmodule SeedTestSetup begin
    using Drill
    using Random

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

    struct NoRNGEnv <: AbstractEnv end
    Drill.observation_space(::NoRNGEnv) = Box(Float32[0.0], Float32[1.0])
    Drill.action_space(::NoRNGEnv) = Box(Float32[-1.0], Float32[1.0])
    Drill.observe(::NoRNGEnv) = Float32[rand()]
    Drill.terminated(::NoRNGEnv) = false
    Drill.truncated(::NoRNGEnv) = false
    Drill.act!(::NoRNGEnv, action) = 0.0f0
    Drill.get_info(::NoRNGEnv) = Dict{String, Any}()
    Drill.reset!(::NoRNGEnv) = nothing
end

@testitem "Random.seed! single and wrappers" tags = [:seeding] setup = [SeedTestSetup] begin
    using Random

    # Single env
    env = SeedTestSetup.DummyEnv(Random.Xoshiro())
    Random.seed!(env, 42)
    reset!(env)
    obs1 = observe(env)
    Random.seed!(env, 42)
    reset!(env)
    obs2 = observe(env)
    @test obs1 == obs2

    # Scaling wrapper over single env
    base = SeedTestSetup.DummyEnv(Random.Xoshiro())
    wrapped = ScalingWrapperEnv(base)
    Random.seed!(wrapped, 123)
    reset!(wrapped)
    w1 = observe(wrapped)
    Random.seed!(wrapped, 123)
    reset!(wrapped)
    w2 = observe(wrapped)
    @test w1 == w2
end

@testitem "Random.seed! parallel envs" tags = [:seeding, :parallel] setup = [SeedTestSetup] begin
    using Random

    # MultiThreadedParallelEnv
    envs_mt = [SeedTestSetup.DummyEnv(Random.Xoshiro()) for _ in 1:3]
    penv_mt = MultiThreadedParallelEnv(envs_mt)
    Random.seed!(penv_mt, 7)
    reset!(penv_mt)
    o1 = observe(penv_mt)
    Random.seed!(penv_mt, 7)
    reset!(penv_mt)
    o2 = observe(penv_mt)
    @test all([o1[i] == o2[i] for i in eachindex(o1)])

    # BroadcastedParallelEnv
    envs_br = [SeedTestSetup.DummyEnv(Random.Xoshiro()) for _ in 1:2]
    penv_br = BroadcastedParallelEnv(envs_br)
    Random.seed!(penv_br, 99)
    reset!(penv_br)
    b1 = observe(penv_br)
    Random.seed!(penv_br, 99)
    reset!(penv_br)
    b2 = observe(penv_br)
    @test all([b1[i] == b2[i] for i in eachindex(b1)])

    # NormalizeWrapperEnv wrapping broadcasted; evaluation mode to avoid stat changes
    nenv = NormalizeWrapperEnv(penv_br; training = false)
    Random.seed!(nenv, 99)
    reset!(nenv)
    n1 = observe(nenv)
    Random.seed!(nenv, 99)
    reset!(nenv)
    n2 = observe(nenv)
    @test all([n1[i] == n2[i] for i in eachindex(n1)])

    # MultiAgentParallelEnv composed of parallel envs
    envs1 = [SeedTestSetup.DummyEnv(Random.Xoshiro()) for _ in 1:2]
    envs2 = [SeedTestSetup.DummyEnv(Random.Xoshiro()) for _ in 1:3]
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

@testitem "Random.seed! no-rng env does not error" tags = [:seeding, :edge] setup = [SeedTestSetup] begin
    using Random
    env = SeedTestSetup.NoRNGEnv()
    Random.seed!(env, 123)  # Should not throw even without rng field
    @test true
end

@testitem "Random.seed! mutates in-place" tags = [:seeding, :inplace] setup = [SeedTestSetup] begin
    using Random

    # Single env: ensure mutation without using return value
    env = SeedTestSetup.DummyEnv(Random.Xoshiro())
    Random.seed!(env, 11)
    reset!(env)
    a1 = observe(env)
    # reseed (no assignment)
    Random.seed!(env, 11)
    reset!(env)
    a2 = observe(env)
    @test a1 == a2

    # ScalingWrapperEnv: underlying env mutated
    base = SeedTestSetup.DummyEnv(Random.Xoshiro())
    wrap = ScalingWrapperEnv(base)
    Random.seed!(wrap, 12)
    reset!(wrap)
    w1 = observe(wrap)
    Random.seed!(wrap, 12) # no assignment
    reset!(wrap)
    w2 = observe(wrap)
    @test w1 == w2

    # BroadcastedParallelEnv: sub-envs seeded with offsets
    envs_b = [SeedTestSetup.DummyEnv(Random.Xoshiro()) for _ in 1:2]
    br = BroadcastedParallelEnv(envs_b)
    Random.seed!(br, 21)
    reset!(br)
    b1 = observe(br)
    # directly check sub-env rngs via repeatability
    Random.seed!(br, 21)
    reset!(br)
    b2 = observe(br)
    @test all([b1[i] == b2[i] for i in eachindex(b1)])

    # MultiThreadedParallelEnv: same check
    envs_mt = [SeedTestSetup.DummyEnv(Random.Xoshiro()) for _ in 1:3]
    mt = MultiThreadedParallelEnv(envs_mt)
    Random.seed!(mt, 33)
    reset!(mt)
    m1 = observe(mt)
    Random.seed!(mt, 33)
    reset!(mt)
    m2 = observe(mt)
    @test all([m1[i] == m2[i] for i in eachindex(m1)])

    # MultiAgentParallelEnv: composed of parallel envs, ensure in-place
    p1 = BroadcastedParallelEnv([SeedTestSetup.DummyEnv(Random.Xoshiro()) for _ in 1:2])
    p2 = MultiThreadedParallelEnv([SeedTestSetup.DummyEnv(Random.Xoshiro()) for _ in 1:1])
    ma = MultiAgentParallelEnv([p1, p2])
    Random.seed!(ma, 44)
    reset!(ma)
    x1 = observe(ma)
    Random.seed!(ma, 44)
    reset!(ma)
    x2 = observe(ma)
    @test all([x1[i] == x2[i] for i in eachindex(x1)])
end
