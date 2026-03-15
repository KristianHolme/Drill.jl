using Test
using Drill
using Random
using Lux
using Lux: AutoZygote, AutoEnzyme, AutoMooncake
using Zygote
using Enzyme
using Mooncake
include("setup.jl")
using .TestSetup

@testset "PPO training with different AD backends (continuous)" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    backends = [
        ("Zygote", AutoZygote()),
        ("Enzyme", AutoEnzyme()),
        ("Enzyme (with runtime activity)", AutoEnzyme(; mode = set_runtime_activity(Reverse))),
    ]

    for (name, ad_backend) in backends
        @testset "$name" begin
            layer = ActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16])
            alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
            agent = Agent(layer, alg; verbose = 0, rng = Random.Xoshiro(42))

            initial_params = deepcopy(agent.train_state.parameters)
            train!(agent, continuous_env, alg, 32; ad_type = ad_backend)
            @test agent.train_state.parameters != initial_params
        end
    end
end

@testset "PPO training with different AD backends (discrete)" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    discrete_obs_space = Drill.Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    discrete_action_space = Drill.Discrete(3, 0)
    discrete_env = BroadcastedParallelEnv(
        [RandomDiscreteEnv(discrete_obs_space, discrete_action_space) for _ in 1:2]
    )

    backends = [
        ("Zygote", AutoZygote()),
        ("Enzyme", AutoEnzyme()),
        ("Enzyme (with runtime activity)", AutoEnzyme(; mode = set_runtime_activity(Reverse))),
    ]

    for (name, ad_backend) in backends
        @testset "$name" begin
            layer = ActorCriticLayer(discrete_obs_space, discrete_action_space; hidden_dims = [16, 16])
            alg = PPO(; n_steps = 8, batch_size = 8, epochs = 2)
            agent = Agent(layer, alg; verbose = 0, rng = Random.Xoshiro(42))

            initial_params = deepcopy(agent.train_state.parameters)
            train!(agent, discrete_env, alg, 32; ad_type = ad_backend)
            @test agent.train_state.parameters != initial_params
        end
    end
end

@testset "SAC training with different AD backends" begin
    continuous_env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
    continuous_obs_space = DrillInterface.observation_space(continuous_env)
    continuous_action_space = DrillInterface.action_space(continuous_env)

    backends = [
        ("Zygote", AutoZygote()),
        ("Enzyme", AutoEnzyme()),
        ("Enzyme (with runtime activity)", AutoEnzyme(; mode = set_runtime_activity(Reverse))),
    ]

    function test_sac_training(ad_backend)
        layer = ContinuousActorCriticLayer(continuous_obs_space, continuous_action_space; hidden_dims = [16, 16], critic_type = QCritic())
        alg = SAC(; start_steps = 4, batch_size = 4)
        agent = Agent(layer, alg; verbose = 0, rng = Random.Xoshiro(42))

        initial_params = deepcopy(agent.train_state.parameters)
        train!(agent, continuous_env, alg, 32; ad_type = ad_backend)
        return agent.train_state.parameters != initial_params
    end
    @testset "$(backends[1][1])" test_sac_training(backends[1][2])
    @testset "$(backends[2][1])" begin
        @test_broken test_sac_training(backends[2][2])
    end
    @testset "$(backends[3][1])" test_sac_training(backends[3][2])
end
