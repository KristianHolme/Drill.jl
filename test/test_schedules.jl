using Test
using Drill
using DrillInterface
using Random
using UnicodePlots: UnicodePlots
using Enzyme
include("setup.jl")
using .TestSetup

const learning_rate = Drill.Algorithms.learning_rate

# The learning rates stored in the optimizer leaves of a learner
etas!(out, leaf::Drill.Algorithms.Optimisers.Leaf) = push!(out, Float32(leaf.rule.eta))
etas!(out, x::Union{Tuple, NamedTuple}) = (foreach(y -> etas!(out, y), x); out)
etas!(out, _) = out
function learner_etas(learner)
    trees = learner isa PPOLearner ? (learner.opt_state,) :
        (learner.actor_opt, learner.critic_opt, learner.ent_opt)
    return unique(etas!(Float32[], trees))
end

@testset "schedule values" begin
    s = LinearSchedule(1.0f-3, 0.0f0)
    @test s isa LinearSchedule{Float32}
    @test LinearSchedule(1, 0) isa LinearSchedule{Float64}
    @test s(0) == 1.0f-3
    @test s(0.5) ≈ 5.0f-4
    @test s(1) == 0

    c = CosineSchedule(1.0, 0.1)
    @test c(0) ≈ 1.0
    @test c(0.5) ≈ 0.55
    @test c(1) ≈ 0.1

    e = ExponentialSchedule(1.0e-2, 1.0e-4)
    @test e(0) ≈ 1.0e-2
    @test e(0.5) ≈ 1.0e-3
    @test e(1) ≈ 1.0e-4
    @test_throws ArgumentError ExponentialSchedule(1.0e-3, 0)

    @test Drill.Algorithms.schedule_value(0.1, 0.7) == 0.1
    @test Drill.Algorithms.schedule_value(p -> 2p, 0.25) == 0.5
    @test sprint(show, s) == "LinearSchedule(0.001, 0.0)"
end

@testset "algorithms accept schedules" begin
    ppo = PPO(; learning_rate = LinearSchedule(1.0e-3, 0))
    @test ppo isa PPO{Float32}
    @test learning_rate(ppo, 0.5) === 5.0f-4
    @test Drill.make_optimizer(ppo).eta === 1.0f-3
    @test learning_rate(PPO(; learning_rate = p -> 1 - p), 0.25) === 0.75f0

    sac = SAC(; learning_rate = CosineSchedule(1.0f-3, 0))
    @test learning_rate(sac, 1) === 0.0f0
    @test Drill.make_optimizer(sac).eta === 1.0f-3
end

@testset "training_progress" begin
    env = BroadcastedParallelEnv([TrackingTargetEnv(8, Random.Xoshiro(i)) for i in 1:2])
    model = ActorCriticModel(observation_space(env), action_space(env); hidden_dims = [8])
    alg = PPO(; n_steps = 8, batch_size = 16, epochs = 1)
    cache = init(RLProblem(env, model), alg; max_steps = 64, verbosity = 0, rng = Random.Xoshiro(1))
    @test training_progress(cache) == 0
    step!(cache)
    @test training_progress(cache) == 0.25
    solve!(cache)
    @test training_progress(cache) == 1
end

@testset "PPO follows the learning rate schedule" begin
    env = BroadcastedParallelEnv([TrackingTargetEnv(8, Random.Xoshiro(i)) for i in 1:2])
    model = ActorCriticModel(observation_space(env), action_space(env); hidden_dims = [8])
    alg = PPO(; n_steps = 8, batch_size = 16, epochs = 1, learning_rate = LinearSchedule(1.0f-3, 0))
    cache = init(RLProblem(env, model), alg; max_steps = 64, verbosity = 0, rng = Random.Xoshiro(1))
    @test learner_etas(cache.learner) == [1.0f-3]
    solve!(cache)
    # The rate is set after each rollout, at progress 1/4, 2/4, 3/4 and 1
    @test cache.stats[:learning_rates] ≈ Float32[7.5f-4, 5.0f-4, 2.5f-4, 0]
    @test learner_etas(cache.learner) == [0.0f0]

    # A zero learning rate leaves the parameters unchanged: the update uses the new rate
    alg = PPO(; n_steps = 8, batch_size = 16, epochs = 1, learning_rate = p -> p > 0 ? 0.0f0 : 1.0f-3)
    cache = init(RLProblem(env, model), alg; max_steps = 64, verbosity = 0, rng = Random.Xoshiro(1))
    ps = deepcopy(Drill.parameters(cache))
    step!(cache)
    @test cache.gradient_updates > 0
    @test Drill.parameters(cache) == ps
end

@testset "constant learning rate leaves the optimizer alone" begin
    env = BroadcastedParallelEnv([TrackingTargetEnv(8, Random.Xoshiro(i)) for i in 1:2])
    model = ActorCriticModel(observation_space(env), action_space(env); hidden_dims = [8])
    alg = PPO(; n_steps = 8, batch_size = 16, epochs = 1, learning_rate = 2.0f-3)
    cache = init(RLProblem(env, model), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(1))
    solve!(cache)
    @test all(==(2.0f-3), cache.stats[:learning_rates])
    @test learner_etas(cache.learner) == [2.0f-3]
end

@testset "SAC follows the learning rate schedule" begin
    env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    model = ContinuousActorCriticModel(
        observation_space(env), action_space(env); hidden_dims = [8], critic_type = QCritic(),
    )
    alg = SAC(; start_steps = 4, batch_size = 4, learning_rate = LinearSchedule(1.0f-3, 0))
    cache = init(RLProblem(env, model), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(1))
    solve!(cache)
    @test issorted(cache.stats[:learning_rates]; rev = true)
    @test last(cache.stats[:learning_rates]) == 0
    @test learner_etas(cache.learner) == [0.0f0]
end

@testset "adjust_learning_rate! and set_learning_rate" begin
    env = BroadcastedParallelEnv([TrackingTargetEnv(8, Random.Xoshiro(i)) for i in 1:2])
    model = ActorCriticModel(observation_space(env), action_space(env); hidden_dims = [8])
    cache = init(RLProblem(env, model), PPO(; n_steps = 8); max_steps = 16, verbosity = 0)
    learner = cache.learner
    copied = set_learning_rate(learner, 1.0f-2)
    @test learner_etas(copied) == [1.0f-2]
    @test learner_etas(learner) == [3.0f-4]
    @test adjust_learning_rate!(learner, 1.0f-2) === learner
    @test learner_etas(learner) == [1.0f-2]
end

@testset "UnicodePlots extension" begin
    @test Base.get_extension(Drill, :Drill_UnicodePlotsExt) !== nothing
    s = CosineSchedule(1.0f-3, 0.0f0)
    @test UnicodePlots.lineplot(s) isa UnicodePlots.Plot
    out = sprint(show, MIME("text/plain"), s)
    @test occursin("CosineSchedule(0.001, 0.0)", out)
    @test occursin("progress", out)
    @test count('\n', out) > 5
end
