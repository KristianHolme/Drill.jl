@testitem "callbacks locals" tags = [:callbacks, :locals] setup = [SharedTestSetup] begin
    using Zygote
    ##
    alg = PPO(; ent_coef = 0.1f0, n_steps = 256, batch_size = 64, epochs = 10)
    env = BroadcastedParallelEnv([SharedTestSetup.CustomEnv() for _ in 1:8])
    env = MonitorWrapperEnv(env)
    env = NormalizeWrapperEnv(env, gamma = alg.gamma)

    policy = ActorCriticLayer(observation_space(env), action_space(env))
    agent = Agent(policy, alg; verbose = 0)

    function test_keys(locals::Dict, keys_to_check::Vector{Symbol})
        for key in keys_to_check
            key_in_locals = haskey(locals, key)
            @test key_in_locals
            if !key_in_locals
                @debug "key $key not in locals"
            end
        end
        true
    end

    @kwdef struct OnTrainingStartCheckLocalsCallback <: AbstractCallback
        keys::Vector{Symbol} = [
            :agent, :env, :alg, :iterations, :total_steps, :max_steps,
            :n_steps, :n_envs, :roll_buffer, :iterations, :total_fps, :callbacks,
        ]
    end
    function Drill.on_training_start(callback::OnTrainingStartCheckLocalsCallback, locals::Dict)
        test_keys(locals, callback.keys)
        true
    end

    @kwdef struct OnRolloutStartCheckLocalsCallback <: AbstractCallback
        first_keys::Vector{Symbol} = [
            :agent, :env, :alg, :iterations, :total_steps,
            :max_steps, :i, :learning_rate,
        ]
        subsequent_keys::Vector{Symbol} = [:agent, :env, :alg, :iterations, :total_steps, :max_steps]
    end
    function Drill.on_rollout_start(callback::OnRolloutStartCheckLocalsCallback, locals::Dict)
        test_keys(locals, callback.first_keys)
        if locals[:i] > 1
            test_keys(locals, callback.subsequent_keys)
        end
        true
    end
    train!(
        agent, env, alg, 3000; callbacks = [
            OnTrainingStartCheckLocalsCallback(),
            OnRolloutStartCheckLocalsCallback(),
        ]
    )
end

@testitem "callbacks early stopping" tags = [:callbacks, :early_stopping] setup = [SharedTestSetup] begin
    using Zygote
    function setup_agent_env_alg()
        alg = PPO(; ent_coef = 0.1f0, n_steps = 64, batch_size = 64, epochs = 10)
        env = BroadcastedParallelEnv([SharedTestSetup.CustomEnv() for _ in 1:8])
        env = MonitorWrapperEnv(env)
        env = NormalizeWrapperEnv(env, gamma = alg.gamma)

        policy = ActorCriticLayer(observation_space(env), action_space(env))
        agent = Agent(policy, alg; verbose = 0)
        return agent, env, alg
    end


    @kwdef struct OnTrainingStartStopEarlyCallback <: AbstractCallback end
    function Drill.on_training_start(callback::OnTrainingStartStopEarlyCallback, locals::Dict)
        return false
    end
    agent, env, alg = setup_agent_env_alg()
    train!(
        agent, env, alg, 3000; callbacks = [
            OnTrainingStartStopEarlyCallback(),
        ]
    )
    @test steps_taken(agent) == 0

    agent, env, alg = setup_agent_env_alg()
    @kwdef struct OnRolloutStartStopEarlyCallback <: AbstractCallback end
    function Drill.on_rollout_start(callback::OnRolloutStartStopEarlyCallback, locals::Dict)
        return false
    end
    train!(agent, env, alg, 3000; callbacks = [OnRolloutStartStopEarlyCallback()])
    @test steps_taken(agent) == 0

    agent, env, alg = setup_agent_env_alg()
    @kwdef struct OnStepStopEarlyCallback <: AbstractCallback
        threshold::Int = 512
    end
    function Drill.on_step(callback::OnStepStopEarlyCallback, locals::Dict)
        continue_training = steps_taken(locals[:agent]) < callback.threshold
        return continue_training
    end
    train!(agent, env, alg, 3000; callbacks = [OnStepStopEarlyCallback(500)])
    @test steps_taken(agent) == 512
end
