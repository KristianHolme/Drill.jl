using Test
using Drill
using DrillInterface
using Zygote
include("setup.jl")
using .TestSetup

@testset "callbacks locals" begin
    alg = PPO(; ent_coef = 0.1f0, n_steps = 256, batch_size = 64, epochs = 10)
    env = BroadcastedParallelEnv([CustomEnv() for _ in 1:8])
    env = MonitorWrapperEnv(env)
    env = NormalizeWrapperEnv(env, gamma = alg.gamma)

    layer = ActorCriticModel(observation_space(env), action_space(env))
    cache = init(RLProblem(env, layer), alg; max_steps = 3000, verbosity = 0)

    function test_cache(cache, keys_to_check::Vector{Symbol})
        for key in keys_to_check
            key_in_cache = hasproperty(cache, key)
            @test key_in_cache
            if !key_in_cache
                @debug "key $key not in cache"
            end
        end
        true
    end

    @kwdef struct OnTrainingStartCheckLocalsCallback <: AbstractCallback
        keys::Vector{Symbol} = [
            :prob, :alg, :model, :adapter, :train_state, :buffer, :logger, :rng,
            :max_steps, :steps_taken, :gradient_updates, :callbacks,
        ]
    end
    function Drill.on_training_start(callback::OnTrainingStartCheckLocalsCallback, cache)
        test_cache(cache, callback.keys)
        true
    end

    @kwdef struct OnRolloutStartCheckLocalsCallback <: AbstractCallback
        first_keys::Vector{Symbol} = [
            :prob, :alg, :model, :adapter, :train_state, :buffer, :logger,
            :max_steps, :steps_taken,
        ]
        subsequent_keys::Vector{Symbol} = [:prob, :alg, :max_steps, :steps_taken]
    end
    function Drill.on_rollout_start(callback::OnRolloutStartCheckLocalsCallback, cache)
        test_cache(cache, callback.first_keys)
        if steps_taken(cache) > 0
            test_cache(cache, callback.subsequent_keys)
        end
        true
    end
    cache.callbacks = [OnTrainingStartCheckLocalsCallback(), OnRolloutStartCheckLocalsCallback()]
    solve!(cache)
end

@testset "callbacks early stopping" begin
    function setup_agent_env_alg()
        alg = PPO(; ent_coef = 0.1f0, n_steps = 64, batch_size = 64, epochs = 10)
        env = BroadcastedParallelEnv([CustomEnv() for _ in 1:8])
        env = MonitorWrapperEnv(env)
        env = NormalizeWrapperEnv(env, gamma = alg.gamma)

        layer = ActorCriticModel(observation_space(env), action_space(env))
        cache = init(RLProblem(env, layer), alg; max_steps = 3000, verbosity = 0)
        return cache, env, alg
    end

    @kwdef struct OnTrainingStartStopEarlyCallback <: AbstractCallback end
    function Drill.on_training_start(callback::OnTrainingStartStopEarlyCallback, cache)
        return false
    end
    cache, env, alg = setup_agent_env_alg()
    cache.callbacks = [OnTrainingStartStopEarlyCallback()]
    solve!(cache)
    @test steps_taken(cache) == 0

    cache, env, alg = setup_agent_env_alg()
    @kwdef struct OnRolloutStartStopEarlyCallback <: AbstractCallback end
    function Drill.on_rollout_start(callback::OnRolloutStartStopEarlyCallback, cache)
        return false
    end
    cache.callbacks = [OnRolloutStartStopEarlyCallback()]
    solve!(cache)
    @test steps_taken(cache) == 0

    cache, env, alg = setup_agent_env_alg()
    @kwdef struct OnStepStopEarlyCallback <: AbstractCallback
        threshold::Int = 512
    end
    function Drill.on_step(callback::OnStepStopEarlyCallback, cache)
        return steps_taken(cache) < callback.threshold
    end
    cache.callbacks = [OnStepStopEarlyCallback(500)]
    solve!(cache)
    @test steps_taken(cache) == 512
end
