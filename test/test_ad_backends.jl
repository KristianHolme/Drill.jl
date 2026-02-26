# Integration tests for different AD backends
@testsnippet ADBackends begin
    using Random
    using Lux: AutoZygote, AutoEnzyme
    using Zygote
    using Enzyme

    env = BroadcastedParallelEnv([SharedTestSetup.CustomEnv(8) for _ in 1:2])
    obs_space = Drill.observation_space(env)
    action_space = Drill.action_space(env)

    backends = [
        ("Zygote", AutoZygote()),
        ("Enzyme", AutoEnzyme()),
        ("Enzyme (with runtime activity)", AutoEnzyme(; mode = set_runtime_activity(Reverse))),
        ("Mooncake", AutoMooncake()),
    ]
end

@testitem "PPO training with different AD backends" tags = [:ppo, :ad_backends] setup = [SharedTestSetup, ADBackends] begin
    for (name, ad_backend) in backends
        @testset "$name" begin
            policy = ActorCriticLayer(obs_space, action_space; hidden_dims = [16, 16])
            alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
            agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42))
            initial_params = deepcopy(agent.train_state.parameters)

            if name == "Enzyme"
                # Known issue: plain AutoEnzyme() can fail here unless runtime activity is enabled.
                changed = try
                    train!(agent, env, alg, 32; ad_type = ad_backend)
                    agent.train_state.parameters != initial_params
                catch
                    false
                end
                @test_broken changed
            else
                train!(agent, env, alg, 32; ad_type = ad_backend)
                @test agent.train_state.parameters != initial_params
            end
        end
    end
end

@testitem "SAC training with different AD backends" tags = [:sac, :ad_backends] setup = [SharedTestSetup, ADBackends] begin
    for (name, ad_backend) in backends
        @testset "$name" begin
            policy = ContinuousActorCriticLayer(obs_space, action_space; hidden_dims = [16, 16], critic_type = QCritic())
            alg = SAC(; start_steps = 4, batch_size = 4)
            agent = Agent(policy, alg; verbose = 0, rng = Random.Xoshiro(42))
            initial_params = deepcopy(agent.train_state.parameters)

            if name == "Enzyme"
                # Known issue: plain AutoEnzyme() can fail here unless runtime activity is enabled.
                changed = try
                    train!(agent, env, alg, 32; ad_type = ad_backend)
                    agent.train_state.parameters != initial_params
                catch
                    false
                end
                @test_broken changed
            else
                train!(agent, env, alg, 32; ad_type = ad_backend)
                @test agent.train_state.parameters != initial_params
            end
        end
    end
end
