# Device support tests: CPU transfer and optional Reactant

@testsnippet DeviceSetup begin
    using Random
    using Lux: AutoZygote, AutoEnzyme, cpu_device
    using Drill
    using Enzyme

    continuous_env = Drill.BroadcastedParallelEnv([SharedTestSetup.CustomEnv(8) for _ in 1:2])
    continuous_obs_space = Drill.observation_space(continuous_env)
    continuous_action_space = Drill.action_space(continuous_env)
end

@testitem "Device transfer with cpu_device (PPO)" tags = [:devices, :ppo] setup = [SharedTestSetup, DeviceSetup] begin
    policy = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42))

    @test Drill.get_device(agent.train_state.parameters) isa typeof(cpu_device())
    agent_on_cpu = agent |> cpu_device()
    @test agent_on_cpu isa Drill.Agent

    initial_params = deepcopy(agent_on_cpu.train_state.parameters)
    train!(agent_on_cpu, continuous_env, alg, 32, ad_type = AutoEnzyme())
    @test agent_on_cpu.train_state.parameters != initial_params
end

@testitem "Device transfer with cpu_device (SAC)" tags = [:devices, :sac] setup = [SharedTestSetup, DeviceSetup] begin
    policy = ContinuousActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16], critic_type = QCritic())
    alg = SAC(; start_steps = 4, batch_size = 4)
    agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42))

    @test Drill.get_device(agent.train_state.parameters) isa typeof(cpu_device())
    agent_on_cpu = agent |> cpu_device()
    @test agent_on_cpu isa Drill.Agent

    initial_params = deepcopy(agent_on_cpu.train_state.parameters)
    train!(agent_on_cpu, continuous_env, alg, 32, ad_type = AutoEnzyme(; mode = set_runtime_activity(Reverse)))
    @test agent_on_cpu.train_state.parameters != initial_params
end

@testitem "Training with Reactant device" tags = [:reactant, :devices] setup = [SharedTestSetup, DeviceSetup] begin
    using Lux
    using Reactant
    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    policy = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42), device)

    # Short training step with Enzyme (recommended with Reactant)
    ad_type = AutoEnzyme()
    train!(agent, continuous_env, alg, 32; ad_type = ad_type)
    @test true  # training proceeded without throw
end

@testitem "PPO constructor builds TrainState on Reactant device without warning" tags = [:reactant, :devices, :ppo] setup = [SharedTestSetup, DeviceSetup] begin
    using Lux
    using Reactant

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    policy = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)

    agent = @test_logs min_level = Base.CoreLogging.Warn begin
        Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42), device)
    end

    @test agent isa Drill.Agent
    @test Drill.get_device(agent.train_state.parameters) !== nothing
    @test isnothing(agent.cache)
end

@testitem "SAC constructor builds TrainState on Reactant device without warning" tags = [:reactant, :devices, :sac] setup = [SharedTestSetup, DeviceSetup] begin
    using Lux
    using Reactant

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    policy = ContinuousActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16], critic_type = QCritic())
    alg = SAC(; start_steps = 4, batch_size = 4)

    agent = @test_logs min_level = Base.CoreLogging.Warn begin
        Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42), device)
    end

    @test agent isa Drill.Agent
    @test Drill.get_device(agent.train_state.parameters) !== nothing
    @test Drill.get_device(agent.aux.Q_target_parameters) !== nothing
    @test isnothing(agent.cache)
end

@testitem "Reactant rollout inference populates and reuses cache" tags = [:reactant, :devices, :ppo] setup = [SharedTestSetup, DeviceSetup] begin
    using Lux
    using Reactant

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    policy = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42), device)
    observations = observe(continuous_env)

    @test Drill.reactant_cache_entry_count(agent) == 0

    actions_1 = predict_actions(agent, observations; deterministic = true, rng = Random.Xoshiro(11))
    cache_size_1 = Drill.reactant_cache_entry_count(agent)

    actions_2 = predict_actions(agent, observations; deterministic = true, rng = Random.Xoshiro(11))
    cache_size_2 = Drill.reactant_cache_entry_count(agent)
    values_only = predict_values(agent, observations)
    cache_size_values = Drill.reactant_cache_entry_count(agent)
    stochastic_actions = predict_actions(agent, observations; deterministic = false, rng = Random.Xoshiro(13))
    cache_size_stochastic = Drill.reactant_cache_entry_count(agent)

    @test !isempty(actions_1)
    @test actions_1 == actions_2
    @test cache_size_1 > 0
    @test cache_size_2 == cache_size_1
    @test length(values_only) == length(observations)
    @test cache_size_values > cache_size_2
    @test length(stochastic_actions) == length(observations)
    @test cache_size_stochastic > cache_size_values

    _, values, logprobs = Drill.get_action_and_values(agent, observations)
    @test length(values) == length(observations)
    @test length(logprobs) == length(observations)
    @test Drill.reactant_cache_entry_count(agent) > cache_size_stochastic
end

@testitem "Reactant deployment inference populates cache and recompiles on shape change" tags = [:reactant, :devices, :ppo] setup = [SharedTestSetup, DeviceSetup] begin
    using Lux
    using Reactant

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    policy = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42), device)
    deployment_policy = extract_policy(agent)
    observations = observe(continuous_env)

    @test Drill.reactant_cache_entry_count(deployment_policy) == 0

    single_action = deployment_policy(observations[1]; deterministic = true, rng = Random.Xoshiro(5))
    cache_size_single = Drill.reactant_cache_entry_count(deployment_policy)
    batch_actions = deployment_policy(observations; deterministic = true, rng = Random.Xoshiro(5))
    cache_size_batch = Drill.reactant_cache_entry_count(deployment_policy)

    @test !isempty(single_action)
    @test length(batch_actions) == length(observations)
    @test cache_size_single > 0
    @test cache_size_batch > cache_size_single
end

@testitem "Reactant SAC inference populates runtime cache" tags = [:reactant, :devices, :sac] setup = [SharedTestSetup, DeviceSetup] begin
    using Lux
    using Reactant

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    policy = ContinuousActorCriticLayer(
        continuous_obs_space,
        continuous_action_space;
        hidden_dims = [16, 16],
        critic_type = QCritic(),
    )
    alg = SAC(; start_steps = 4, batch_size = 4)
    agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42), device)
    observations = observe(continuous_env)

    @test Drill.reactant_cache_entry_count(agent) == 0

    actions = predict_actions(agent, observations; deterministic = true, rng = Random.Xoshiro(17))
    cache_size = Drill.reactant_cache_entry_count(agent)

    @test length(actions) == length(observations)
    @test cache_size > 0
end

@testitem "Reactant cache invalidates on device adaptation" tags = [:reactant, :devices, :ppo] setup = [SharedTestSetup, DeviceSetup] begin
    using Lux
    using Reactant

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    policy = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42), device)
    observations = observe(continuous_env)

    predict_actions(agent, observations; deterministic = true, rng = Random.Xoshiro(7))
    @test Drill.reactant_cache_entry_count(agent) > 0

    agent_cpu = agent |> cpu_device()
    @test Drill.reactant_cache_entry_count(agent_cpu) == 0

    deployment_policy = extract_policy(agent)
    deployment_policy(observations; deterministic = true, rng = Random.Xoshiro(7))
    @test Drill.reactant_cache_entry_count(deployment_policy) > 0

    deployment_policy_cpu = deployment_policy |> cpu_device()
    @test Drill.reactant_cache_entry_count(deployment_policy_cpu) == 0
end

@testitem "Reactant cache invalidates after loading policy state" tags = [:reactant, :devices, :ppo] setup = [SharedTestSetup, DeviceSetup] begin
    using Lux
    using Reactant

    Reactant.set_default_backend("cpu")
    device = Lux.reactant_device()
    policy = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42), device)
    observations = observe(continuous_env)

    predict_actions(agent, observations; deterministic = true, rng = Random.Xoshiro(19))
    @test Drill.reactant_cache_entry_count(agent) > 0

    mktempdir() do dir
        saved_path = save_policy_params_and_state(agent, joinpath(dir, "ppo_agent"))
        load_policy_params_and_state!(agent, alg, saved_path)
        @test Drill.reactant_cache_entry_count(agent) == 0
        @test Drill.get_device(agent.train_state.parameters) !== nothing
    end
end
