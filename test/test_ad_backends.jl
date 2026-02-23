# Integration tests for different AD backends
@testsnippet ADBackends begin
    using Random
    using Lux: AutoZygote, AutoEnzyme, AutoMooncake
    using Zygote
    using Enzyme
    using Mooncake

    continuous_env = Drill.BroadcastedParallelEnv([SharedTestSetup.CustomEnv(8) for _ in 1:2])
    continuous_obs_space = Drill.observation_space(continuous_env)
    continuous_action_space = Drill.action_space(continuous_env)

    discrete_obs_space = Drill.Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    discrete_action_space = Drill.Discrete(3, 0)
    discrete_env = Drill.BroadcastedParallelEnv(
        [SharedTestSetup.RandomDiscreteEnv(discrete_obs_space, discrete_action_space) for _ in 1:2]
    )

    backends = [
        ("Zygote", AutoZygote()),
        ("Enzyme", AutoEnzyme()),
        ("Enzyme (with runtime activity)", AutoEnzyme(; mode = set_runtime_activity(Reverse))),
        # ("Mooncake", AutoMooncake()),
    ]
end

@testitem "PPO training with different AD backends (continuous)" tags = [:ppo, :ad_backends] setup = [SharedTestSetup, ADBackends] begin
    for (name, ad_backend) in backends
        @testset "$name" begin
            policy = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
            alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
            agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42))

            initial_params = deepcopy(agent.train_state.parameters)
            train!(agent, continuous_env, alg, 32; ad_type = ad_backend)
            @test agent.train_state.parameters != initial_params
        end
    end
end

@testitem "PPO training with different AD backends (discrete)" tags = [:ppo, :ad_backends] setup = [SharedTestSetup, ADBackends] begin
    for (name, ad_backend) in backends
        @testset "$name" begin
            policy = ActorCriticLayer(discrete_obs_space, discrete_action_space; hidden_dims = [16, 16])
            alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
            agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42))

            initial_params = deepcopy(agent.train_state.parameters)
            train!(agent, discrete_env, alg, 32; ad_type = ad_backend)
            @test agent.train_state.parameters != initial_params
        end
    end
end

@testitem "SAC training with different AD backends" tags = [:sac, :ad_backends] setup = [SharedTestSetup, ADBackends] begin
    for (name, ad_backend) in backends
        @testset "$name" begin
            policy = ContinuousActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16], critic_type = QCritic())
            alg = SAC(; start_steps = 4, batch_size = 4)
            agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42))

            initial_params = deepcopy(agent.train_state.parameters)
            train!(agent, continuous_env, alg, 32; ad_type = ad_backend)
            @test agent.train_state.parameters != initial_params
        end
    end
end
