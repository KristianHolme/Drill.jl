module BenchUtils

using DRiL
using ClassicControlEnvironments
using Random

const DEFAULT_SEED = 42
const DEFAULT_N_ENVS = 2
const DEFAULT_ROLLOUT_STEPS = 64
const DEFAULT_TRAIN_STEPS = 256
const DEFAULT_SAMPLES = 1000

function make_cartpole_env(; n_envs::Int = DEFAULT_N_ENVS, seed::Int = DEFAULT_SEED)
    envs = [CartPoleEnv(; rng = Random.Xoshiro(seed + i)) for i in 1:n_envs]
    env = BroadcastedParallelEnv(envs)
    return env
end

function make_pendulum_env(; n_envs::Int = DEFAULT_N_ENVS, seed::Int = DEFAULT_SEED)
    envs = [PendulumEnv(; rng = Random.Xoshiro(seed + i)) for i in 1:n_envs]
    env = BroadcastedParallelEnv(envs)
    return env
end

function make_ppo_agent(env::AbstractParallelEnv; seed::Int = DEFAULT_SEED)
    rng = Random.Xoshiro(seed)
    alg = PPO(; n_steps = 32, batch_size = 32, epochs = 1, learning_rate = 1.0f-3)
    layer = ActorCriticLayer(
        observation_space(env),
        action_space(env);
        hidden_dims = [32, 32]
    )
    agent = Agent(layer, alg; verbose = 0, logger = NoTrainingLogger(), rng = rng)
    return agent, alg
end

function make_sac_agent(env::AbstractParallelEnv; seed::Int = DEFAULT_SEED)
    rng = Random.Xoshiro(seed)
    alg = SAC(;
        buffer_capacity = 1_000,
        batch_size = 32,
        start_steps = 10,
        train_freq = 1,
        gradient_steps = 1,
    )
    layer = SACLayer(
        observation_space(env),
        action_space(env);
        hidden_dims = [32, 32]
    )
    agent = Agent(layer, alg; verbose = 0, logger = NoTrainingLogger(), rng = rng)
    return agent, alg
end

function setup_rollout_collection(; n_envs::Int = DEFAULT_N_ENVS, n_steps::Int = DEFAULT_ROLLOUT_STEPS)
    env = make_cartpole_env(; n_envs = n_envs)
    agent, alg = make_ppo_agent(env)
    buffer = RolloutBuffer(
        observation_space(env),
        action_space(env),
        alg.gae_lambda,
        alg.gamma,
        n_steps,
        n_envs,
    )
    reset!(env)
    return env, agent, alg, buffer
end

function setup_replay_collection(; n_envs::Int = DEFAULT_N_ENVS, n_steps::Int = DEFAULT_ROLLOUT_STEPS)
    env = make_pendulum_env(; n_envs = n_envs)
    agent, alg = make_sac_agent(env)
    buffer = ReplayBuffer(observation_space(env), action_space(env), 1_000)
    reset!(env)
    return env, agent, alg, buffer, n_steps
end

function setup_training_ppo(; n_envs::Int = DEFAULT_N_ENVS, max_steps::Int = DEFAULT_TRAIN_STEPS)
    env = make_cartpole_env(; n_envs = n_envs)
    agent, alg = make_ppo_agent(env)
    reset!(env)
    return env, agent, alg, max_steps
end

function setup_training_sac(; n_envs::Int = DEFAULT_N_ENVS, max_steps::Int = DEFAULT_TRAIN_STEPS)
    env = make_pendulum_env(; n_envs = n_envs)
    agent, alg = make_sac_agent(env)
    reset!(env)
    return env, agent, alg, max_steps
end

function setup_wrapper_envs(; n_envs::Int = DEFAULT_N_ENVS)
    base_env = make_cartpole_env(; n_envs = n_envs)
    monitor_env = MonitorWrapperEnv(make_cartpole_env(; n_envs = n_envs))
    normalize_env = NormalizeWrapperEnv(make_cartpole_env(; n_envs = n_envs), gamma = 0.99f0)
    reset!(base_env)
    reset!(monitor_env)
    reset!(normalize_env)
    rng = Random.Xoshiro(DEFAULT_SEED)
    actions = rand(rng, action_space(base_env), n_envs)
    return base_env, monitor_env, normalize_env, actions
end

function setup_threaded_envs(; n_envs::Int = DEFAULT_N_ENVS)
    envs = [CartPoleEnv(; rng = Random.Xoshiro(DEFAULT_SEED + i)) for i in 1:n_envs]
    threaded_env = MultiThreadedParallelEnv(envs)
    reset!(threaded_env)
    rng = Random.Xoshiro(DEFAULT_SEED)
    actions = rand(rng, action_space(threaded_env), n_envs)
    return threaded_env, actions
end

end
