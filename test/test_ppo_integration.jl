using Test
using Drill
using Random
using Zygote
include("setup.jl")
using .TestSetup

@testset "PPO smoke test improves tracking env performance" begin
    function make_parallel_env(seed::Int, n_envs::Int)
        envs = [TrackingTargetEnv(16, Random.MersenneTwister(seed + i)) for i in 1:n_envs]
        penv = BroadcastedParallelEnv(envs)
        Random.seed!(penv, seed)
        return penv
    end

    n_envs = 4
    train_env = make_parallel_env(1_234, n_envs)
    baseline_env = make_parallel_env(9_999, n_envs)
    trained_eval_env = make_parallel_env(9_999, n_envs)

    obs_space = Drill.observation_space(train_env)
    act_space = Drill.action_space(train_env)

    layer = ActorCriticLayer(obs_space, act_space; hidden_dims = [64, 64])
    alg = PPO(; n_steps = 64, batch_size = 32, epochs = 10, learning_rate = 3.0f-3)
    agent = Agent(layer, alg; verbose = 0, rng = Random.MersenneTwister(42), logger = NoTrainingLogger())

    baseline_stats = evaluate_agent(agent, baseline_env; n_eval_episodes = 64, deterministic = true, warn = false)
    baseline_mean_step = baseline_stats.mean_reward / baseline_stats.mean_length
    @test baseline_mean_step < 0.6f0

    max_steps = alg.n_steps * n_envs * 50
    train!(agent, train_env, alg, max_steps)

    trained_stats = evaluate_agent(agent, trained_eval_env; n_eval_episodes = 64, deterministic = true, warn = false)

    trained_mean_step = trained_stats.mean_reward / trained_stats.mean_length
    @test trained_mean_step > baseline_mean_step + 0.2f0
    @test trained_mean_step > 0.75f0
end

@testset "PPO agent serialization roundtrip" begin
    function make_parallel_env(seed::Int, n_envs::Int)
        envs = [TrackingTargetEnv(16, Random.MersenneTwister(seed + i)) for i in 1:n_envs]
        penv = MultiThreadedParallelEnv(envs)
        Random.seed!(penv, seed)
        return penv
    end

    n_envs = 2
    train_env = make_parallel_env(321, n_envs)

    obs_space = Drill.observation_space(train_env)
    act_space = Drill.action_space(train_env)

    layer = ActorCriticLayer(obs_space, act_space; hidden_dims = [32, 32])
    alg = PPO(; n_steps = 16, batch_size = 16, epochs = 2, learning_rate = 5.0f-4)
    agent = Agent(layer, alg; verbose = 0, rng = Random.MersenneTwister(77), logger = NoTrainingLogger())

    max_steps = alg.n_steps * n_envs * 10
    train!(agent, train_env, alg, max_steps)

    mktempdir() do dir
        path = joinpath(dir, "ppo_agent")
        saved_path = save_layer_params_and_state(agent, path)

        new_layer = ActorCriticLayer(obs_space, act_space; hidden_dims = [32, 32])
        new_agent = Agent(new_layer, alg; verbose = 0, rng = Random.MersenneTwister(88), logger = NoTrainingLogger())

        load_layer_params_and_state!(new_agent, alg, saved_path)

        @test new_agent.train_state.parameters == agent.train_state.parameters
        @test new_agent.train_state.states == agent.train_state.states

        observations = [rand(Random.MersenneTwister(42), obs_space) for _ in 1:5]
        original_actions = predict_actions(agent, observations; deterministic = true, rng = Random.MersenneTwister(999))
        loaded_actions = predict_actions(new_agent, observations; deterministic = true, rng = Random.MersenneTwister(999))
        @test original_actions == loaded_actions
    end
end
