using Test
using Drill
using DrillInterface
using Random
using Lux
using Lux: cpu_device
using Enzyme
using Zygote
using Reactant
include("setup.jl")
using .TestSetup

@testset "Device transfer with cpu_device (PPO)" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    layer = ActorCriticModel(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    cache = init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42))

    @test Drill.get_device(Drill.parameters(cache)) isa typeof(cpu_device())
    cache_on_cpu = cache |> cpu_device()
    @test cache_on_cpu isa Drill.RLCache

    initial_params = deepcopy(Drill.parameters(cache_on_cpu))
    cache_on_cpu.ad_type = AutoEnzyme()
    solve!(cache_on_cpu)
    @test Drill.parameters(cache_on_cpu) != initial_params
end

@testset "Device transfer with cpu_device (SAC)" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    layer = ContinuousActorCriticModel(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16], critic_type = QCritic())
    alg = SAC(; start_steps = 4, batch_size = 4)
    cache = init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42))

    @test Drill.get_device(Drill.parameters(cache)) isa typeof(cpu_device())
    cache_on_cpu = cache |> cpu_device()
    @test cache_on_cpu isa Drill.RLCache

    initial_params = deepcopy(Drill.parameters(cache_on_cpu))
    cache_on_cpu.ad_type = AutoEnzyme(; mode = set_runtime_activity(Reverse))
    solve!(cache_on_cpu)
    @test Drill.parameters(cache_on_cpu) != initial_params
end

@testset "Training with Reactant device" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ActorCriticModel(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    cache = init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)

    cache.ad_type = AutoEnzyme()
    initial_params = deepcopy(cpu_device()(Drill.parameters(cache)))
    solve!(cache)
    @test Drill.get_device(Drill.parameters(cache)) isa Lux.ReactantDevice
    @test cpu_device()(Drill.parameters(cache)) != initial_params
    @test cache.gradient_updates > 0
    # inference kernels plus the compiled update
    @test Drill.reactant_cache_entry_count(cache) > 0
end

@testset "SAC training with Reactant device" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ContinuousActorCriticModel(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16], critic_type = QCritic())
    alg = SAC(; start_steps = 4, batch_size = 4, target_update_interval = 2)
    cache = init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)
    cache.ad_type = AutoEnzyme()

    initial_params = deepcopy(cpu_device()(Drill.parameters(cache)))
    initial_target = deepcopy(cpu_device()(cache.learner.target_ps))
    solve!(cache)
    @test cache.learner isa Drill.SACLearner
    @test Drill.get_device(Drill.parameters(cache)) isa Lux.ReactantDevice
    @test cache.gradient_updates > 0
    @test cpu_device()(Drill.parameters(cache)) != initial_params
    @test cpu_device()(cache.learner.target_ps) != initial_target
    @test all(isfinite, cache.stats[:critic_losses])
    @test Drill.reactant_cache_entry_count(cache) > 0
end

@testset "PPO update parity: CPU vs compiled on Reactant" begin
    env = BroadcastedParallelEnv([CustomEnv(8, Random.Xoshiro(i)) for i in 1:2])
    layer = ActorCriticModel(observation_space(env), action_space(env); hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 16, epochs = 1)
    cache = init(RLProblem(env, layer), alg; max_steps = 16, verbosity = 0, rng = Random.Xoshiro(42))
    Drill.collect_rollout!(cache.buffer, cache, alg, env)
    Drill.prepare_rollout!(cache.buffer, alg)
    b = cache.buffer
    batch = (
        b.observations, Drill.prepare_training_actions(b.actions, action_space(b)),
        b.advantages, b.returns, b.logprobs, b.values,
    )
    initial = deepcopy(cache.learner)
    cpu_learner, cpu_metrics = ppo_update(alg, cache.model, deepcopy(initial), batch, AutoZygote())

    Reactant.set_default_backend("cpu")
    dev = Lux.reactant_device()
    r_cache = init(RLProblem(env, layer), alg; max_steps = 16, verbosity = 0, rng = Random.Xoshiro(42), device = dev)
    r_learner = Drill.init_learner(
        alg, cache.model, dev(initial.ps), dev(initial.st); device = dev,
    )
    ad = Drill.gradient_backend(dev, AutoEnzyme())
    new_r_learner, r_metrics = Drill.run_update(dev, r_cache, ppo_update, alg, cache.model, r_learner, dev(batch), ad)
    r_metrics = Drill.host_metrics(r_metrics)
    @test Drill.reactant_cache_entry_count(r_cache) == 1

    for name in keys(cpu_metrics)
        @test isapprox(r_metrics[name], cpu_metrics[name]; rtol = 1.0e-4, atol = 1.0e-6)
    end
    cpu_ps = cpu_learner.ps
    r_ps = cpu_device()(new_r_learner.ps)
    @test Lux.Functors.fleaves(r_ps) |> length == Lux.Functors.fleaves(cpu_ps) |> length
    for (r, c) in zip(Lux.Functors.fleaves(r_ps), Lux.Functors.fleaves(cpu_ps))
        @test isapprox(r, c; rtol = 1.0e-4)
    end
    # The parameter step itself agrees, not only the (nearly equal) parameters
    init_ps = initial.ps
    for (r, c, p) in zip(Lux.Functors.fleaves(r_ps), Lux.Functors.fleaves(cpu_ps), Lux.Functors.fleaves(init_ps))
        @test isapprox(r .- p, c .- p; rtol = 1.0e-2, atol = 1.0e-6)
    end

    # Running the compiled update again reuses the cached executable
    Drill.run_update(dev, r_cache, ppo_update, alg, cache.model, new_r_learner, dev(batch), ad)
    @test Drill.reactant_cache_entry_count(r_cache) == 1
end

@testset "PPO constructor builds learner on Reactant device without warning" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ActorCriticModel(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)

    cache = @test_logs min_level = Base.CoreLogging.Warn begin
        init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)
    end

    @test cache isa Drill.RLCache
    @test cache.learner isa Drill.PPOLearner
    @test Drill.get_device(Drill.parameters(cache)) !== nothing
    @test isnothing(cache.inference_cache)
end

@testset "SAC constructor builds learner on Reactant device without warning" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ContinuousActorCriticModel(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16], critic_type = QCritic())
    alg = SAC(; start_steps = 4, batch_size = 4)

    cache = @test_logs min_level = Base.CoreLogging.Warn begin
        init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)
    end

    @test cache isa Drill.RLCache
    @test cache.learner isa Drill.SACLearner
    @test Drill.get_device(Drill.parameters(cache)) !== nothing
    @test Drill.get_device(cache.learner.target_ps) isa Lux.ReactantDevice
    @test isnothing(cache.inference_cache)
end

@testset "Reactant rollout inference populates and reuses cache" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ActorCriticModel(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    cache = init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)
    observations = observe(continuous_env)

    @test Drill.reactant_cache_entry_count(cache) == 0

    actions_1 = predict_actions(cache, observations; deterministic = true, rng = Random.Xoshiro(11))
    cache_size_1 = Drill.reactant_cache_entry_count(cache)

    actions_2 = predict_actions(cache, observations; deterministic = true, rng = Random.Xoshiro(11))
    cache_size_2 = Drill.reactant_cache_entry_count(cache)
    values_only = predict_values(cache, observations)
    cache_size_values = Drill.reactant_cache_entry_count(cache)
    stochastic_actions = predict_actions(cache, observations; deterministic = false, rng = Random.Xoshiro(13))
    cache_size_stochastic = Drill.reactant_cache_entry_count(cache)

    @test !isempty(actions_1)
    @test actions_1 == actions_2
    @test cache_size_1 > 0
    @test cache_size_2 == cache_size_1
    @test length(values_only) == size(observations, 2)
    @test cache_size_values > cache_size_2
    @test length(stochastic_actions) == size(observations, 2)
    @test cache_size_stochastic > cache_size_values

    _, values, logprobs = Drill.get_action_and_values(cache, observations)
    @test length(values) == size(observations, 2)
    @test length(logprobs) == size(observations, 2)
    @test Drill.reactant_cache_entry_count(cache) > cache_size_stochastic
end

@testset "Reactant deployment inference populates cache and recompiles on shape change" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ActorCriticModel(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    cache = init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)
    deployment_layer = extract_policy(cache)
    observations = observe(continuous_env)

    @test Drill.reactant_cache_entry_count(deployment_layer) == 0

    single_action = deployment_layer(observations[:, 1]; deterministic = true, rng = Random.Xoshiro(5))
    cache_size_single = Drill.reactant_cache_entry_count(deployment_layer)
    batch_actions = deployment_layer(observations; deterministic = true, rng = Random.Xoshiro(5))
    cache_size_batch = Drill.reactant_cache_entry_count(deployment_layer)

    @test !isempty(single_action)
    @test length(batch_actions) == size(observations, 2)
    @test cache_size_single > 0
    @test cache_size_batch > cache_size_single
end

@testset "Reactant SAC inference populates runtime cache" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ContinuousActorCriticModel(
        continuous_obs_space,
        continuous_action_space;
        hidden_dims = [16, 16],
        critic_type = QCritic(),
    )
    alg = SAC(; start_steps = 4, batch_size = 4)
    cache = init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)
    observations = observe(continuous_env)

    @test Drill.reactant_cache_entry_count(cache) == 0

    actions = predict_actions(cache, observations; deterministic = true, rng = Random.Xoshiro(17))
    cache_size = Drill.reactant_cache_entry_count(cache)

    @test length(actions) == size(observations, 2)
    @test cache_size > 0
end

@testset "Reactant cache invalidates on device adaptation" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ActorCriticModel(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    cache = init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)
    observations = observe(continuous_env)

    predict_actions(cache, observations; deterministic = true, rng = Random.Xoshiro(7))
    @test Drill.reactant_cache_entry_count(cache) > 0

    cache_cpu = cache |> cpu_device()
    @test Drill.reactant_cache_entry_count(cache_cpu) == 0

    deployment_layer = extract_policy(cache)
    deployment_layer(observations; deterministic = true, rng = Random.Xoshiro(7))
    @test Drill.reactant_cache_entry_count(deployment_layer) > 0

    deployment_policy_cpu = deployment_layer |> cpu_device()
    @test Drill.reactant_cache_entry_count(deployment_policy_cpu) == 0
end

@testset "Reactant cache invalidates after loading layer state" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ActorCriticModel(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    cache = init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)
    observations = observe(continuous_env)

    predict_actions(cache, observations; deterministic = true, rng = Random.Xoshiro(19))
    @test Drill.reactant_cache_entry_count(cache) > 0

    mktempdir() do dir
        saved_path = save_model_params_and_state(cache, joinpath(dir, "ppo_agent"))
        load_model_params_and_state!(cache, alg, saved_path)
        @test Drill.reactant_cache_entry_count(cache) == 0
        @test Drill.get_device(Drill.parameters(cache)) !== nothing
    end
end
