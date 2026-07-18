using Test
using Drill
using DrillInterface
using Random
using SciMLBase: ReturnCode
using Zygote
include("setup.jl")
using .TestSetup

@testset "normalize_verbosity defaults and Int shorthand" begin
    @test normalize_verbosity((;)) == Verbosity(; meter = 2, table = false, timer = true)
    @test normalize_verbosity((; meter = 1)) == Verbosity(; meter = 1, table = false, timer = true)
    @test normalize_verbosity((; table = true, timer = false)) ==
        Verbosity(; meter = 2, table = true, timer = false)

    @test normalize_verbosity(0) == Verbosity(; meter = 0, table = false, timer = false)
    @test normalize_verbosity(1) == Verbosity(; meter = 1, table = false, timer = false)
    @test normalize_verbosity(2) == Verbosity(; meter = 2, table = false, timer = false)

    @test normalize_verbosity(Verbosity(; meter = 1, table = true, timer = false)) ==
        Verbosity(; meter = 1, table = true, timer = false)
end

@testset "normalize_verbosity clamps meter > 2 with warning" begin
    v = @test_logs (:warn, r"max level is 2") normalize_verbosity(3)
    @test v == Verbosity(; meter = 2, table = false, timer = false)

    v2 = @test_logs (:warn, r"max level is 2") normalize_verbosity((; meter = 5, table = true))
    @test v2 == Verbosity(; meter = 2, table = true, timer = true)
end

@testset "normalize_verbosity rejects negative meter" begin
    @test_throws ArgumentError normalize_verbosity(-1)
    @test_throws ArgumentError normalize_verbosity((; meter = -2))
end

@testset "print_training_table without extension errors" begin
    ext = Base.get_extension(Drill, :Drill_PrettyTablesExt)
    if ext === nothing
        n_envs = 1
        env = BroadcastedParallelEnv([TrackingTargetEnv(8, Random.Xoshiro(1))])
        layer = ActorCriticModel(observation_space(env), action_space(env); hidden_dims = [16])
        cache = init(
            RLProblem(env, layer),
            PPO(; n_steps = 8, batch_size = 8, epochs = 1);
            max_steps = 8,
            verbosity = 0,
        )
        @test_throws ArgumentError print_training_table(cache)
    else
        @test true
    end
end

@testset "training smoke with meter=2" begin
    n_envs = 2
    envs = [TrackingTargetEnv(8, Random.Xoshiro(10 + i)) for i in 1:n_envs]
    env = BroadcastedParallelEnv(envs)
    layer = ActorCriticModel(observation_space(env), action_space(env); hidden_dims = [32])
    alg = PPO(; n_steps = 16, batch_size = 16, epochs = 1, learning_rate = 3.0f-3)
    max_steps = alg.n_steps * n_envs * 2

    cache = init(
        RLProblem(env, layer),
        alg;
        max_steps,
        verbosity = (; meter = 2, table = false, timer = false),
        rng = Random.Xoshiro(42),
    )
    @test cache.verbosity.meter == 2
    @test cache.verbosity.table == false
    @test cache.verbosity.timer == false
    @test cache.progress_meter !== nothing

    redirect_stderr(devnull) do
        solve!(cache)
    end
    @test cache.retcode == ReturnCode.Success
    @test cache.steps_taken >= max_steps
end

@testset "PrettyTables extension table=true" begin
    try
        @eval Main using PrettyTables
    catch
        @test_skip "PrettyTables not available"
        return
    end

    # Re-load Drill side so the extension can activate in this process if needed.
    ext = Base.get_extension(Drill, :Drill_PrettyTablesExt)
    if ext === nothing
        @test_skip "Drill_PrettyTablesExt not loaded"
        return
    end

    n_envs = 2
    envs = [TrackingTargetEnv(8, Random.Xoshiro(20 + i)) for i in 1:n_envs]
    env = BroadcastedParallelEnv(envs)
    layer = ActorCriticModel(observation_space(env), action_space(env); hidden_dims = [32])
    alg = PPO(; n_steps = 16, batch_size = 16, epochs = 1)
    max_steps = alg.n_steps * n_envs

    cache = init(
        RLProblem(env, layer),
        alg;
        max_steps,
        verbosity = (; meter = 0, table = true, timer = false),
        rng = Random.Xoshiro(7),
    )
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            solve!(cache)
        end
    end
    @test cache.steps_taken >= max_steps
end
