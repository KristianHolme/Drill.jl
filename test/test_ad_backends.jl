# Integration tests for different AD backends

@testitem "PPO training with different AD backends" tags = [:ppo, :ad_backends] setup = [SharedTestSetup] begin
    using Random
    using Lux: AutoZygote, AutoEnzyme
    using Zygote
    using Enzyme

    env = BroadcastedParallelEnv([SharedTestSetup.CustomEnv(8) for _ in 1:2])
    obs_space = DRiL.observation_space(env)
    action_space = DRiL.action_space(env)

    backends = [
        ("Zygote", AutoZygote()),
        ("Enzyme", AutoEnzyme()),
    ]

    for (name, ad_backend) in backends
        @testset "$name" begin
            policy = ActorCriticLayer(obs_space, action_space; hidden_dims = [16, 16])
            alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
            agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42))

            initial_params = deepcopy(agent.train_state.parameters)
            train!(agent, env, alg, 32; ad_type = ad_backend)
            @test agent.train_state.parameters != initial_params
        end
    end
end

@testitem "SAC training with different AD backends" tags = [:sac, :ad_backends] setup = [SharedTestSetup] begin
    using Random
    using Lux: AutoZygote, AutoEnzyme
    using Zygote
    using Enzyme

    env = BroadcastedParallelEnv([SharedTestSetup.CustomEnv(8) for _ in 1:2])
    obs_space = DRiL.observation_space(env)
    action_space = DRiL.action_space(env)

    backends = [
        ("Zygote", AutoZygote()),
        ("Enzyme", AutoEnzyme()),
    ]

    for (name, ad_backend) in backends
        @testset "$name" begin
            policy = ContinuousActorCriticLayer(obs_space, action_space; hidden_dims = [16, 16], critic_type = QCritic())
            alg = SAC(; start_steps = 4, batch_size = 4)
            agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42))

            initial_params = deepcopy(agent.train_state.parameters)
            train!(agent, env, alg, 32; ad_type = ad_backend)
            @test agent.train_state.parameters != initial_params
        end
    end
end
