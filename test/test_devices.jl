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
    policy = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
    agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42))
    agent = agent |> Lux.reactant_device()

    # Short training step with Enzyme (recommended with Reactant)
    ad_type = AutoEnzyme()
    train!(agent, continuous_env, alg, 32; ad_type = ad_type)
    @test true  # training proceeded without throw
end
