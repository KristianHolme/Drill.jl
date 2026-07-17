using Test
using Drill
using DrillInterface
using Random
using Lux
using Lux: cpu_device
using Enzyme
using Reactant
include("setup.jl")
using .TestSetup

@testset "Device transfer with cpu_device (PPO)" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    layer = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
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

    layer = ContinuousActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16], critic_type = QCritic())
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
    layer = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    cache = init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)

    ad_type = AutoEnzyme()
    cache.ad_type = ad_type
    solve!(cache)
    @test true
end

# Full SAC train! on Reactant still requires compiling host-side target-Q / entropy
# forwards (outside Lux TrainState). Constructor + inference coverage is below.

@testset "PPO constructor builds TrainState on Reactant device without warning" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)

    cache = @test_logs min_level = Base.CoreLogging.Warn begin
        init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)
    end

    @test cache isa Drill.RLCache
    @test cache.train_state isa Drill.PPOTrainState
    @test Drill.get_device(Drill.parameters(cache)) !== nothing
    @test isnothing(cache.inference_cache)
end

@testset "SAC constructor builds TrainState on Reactant device without warning" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ContinuousActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16], critic_type = QCritic())
    alg = SAC(; start_steps = 4, batch_size = 4)

    cache = @test_logs min_level = Base.CoreLogging.Warn begin
        init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)
    end

    @test cache isa Drill.RLCache
    @test cache.train_state isa Drill.SACTrainState
    @test Drill.get_device(Drill.parameters(cache)) !== nothing
    @test Drill.get_device(cache.train_state.target_parameters) !== nothing
    @test isnothing(cache.inference_cache)
end

@testset "Reactant rollout inference populates and reuses cache" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
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
    @test length(values_only) == length(observations)
    @test cache_size_values > cache_size_2
    @test length(stochastic_actions) == length(observations)
    @test cache_size_stochastic > cache_size_values

    _, values, logprobs = Drill.get_action_and_values(cache, observations)
    @test length(values) == length(observations)
    @test length(logprobs) == length(observations)
    @test Drill.reactant_cache_entry_count(cache) > cache_size_stochastic
end

@testset "Reactant deployment inference populates cache and recompiles on shape change" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    cache = init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)
    deployment_layer = extract_policy(cache)
    observations = observe(continuous_env)

    @test Drill.reactant_cache_entry_count(deployment_layer) == 0

    single_action = deployment_layer(observations[1]; deterministic = true, rng = Random.Xoshiro(5))
    cache_size_single = Drill.reactant_cache_entry_count(deployment_layer)
    batch_actions = deployment_layer(observations; deterministic = true, rng = Random.Xoshiro(5))
    cache_size_batch = Drill.reactant_cache_entry_count(deployment_layer)

    @test !isempty(single_action)
    @test length(batch_actions) == length(observations)
    @test cache_size_single > 0
    @test cache_size_batch > cache_size_single
end

@testset "Reactant SAC inference populates runtime cache" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ContinuousActorCriticLayer(
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

    @test length(actions) == length(observations)
    @test cache_size > 0
end

@testset "Reactant cache invalidates on device adaptation" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    layer = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
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
    layer = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    cache = init(RLProblem(continuous_env, layer), alg; max_steps = 32, verbosity = 0, rng = Random.Xoshiro(42), device)
    observations = observe(continuous_env)

    predict_actions(cache, observations; deterministic = true, rng = Random.Xoshiro(19))
    @test Drill.reactant_cache_entry_count(cache) > 0

    mktempdir() do dir
        saved_path = save_layer_params_and_state(cache, joinpath(dir, "ppo_agent"))
        load_layer_params_and_state!(cache, alg, saved_path)
        @test Drill.reactant_cache_entry_count(cache) == 0
        @test Drill.get_device(Drill.parameters(cache)) !== nothing
    end
end
