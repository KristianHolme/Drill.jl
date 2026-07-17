module BenchUtils

using Drill
using Drill.Lux
using ClassicControlEnvironments
using Random
using Statistics: mean

# MLUtils is a Drill dependency; access DataLoader via the Solve submodule.
const DataLoader = Drill.Solve.DataLoader

const DEFAULT_SEED = 42
const DEFAULT_N_ENVS = 2
const DEFAULT_ROLLOUT_STEPS = 32
const DEFAULT_TRAIN_STEPS = 64
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

function make_ppo_cache(env::AbstractParallelEnv; seed::Int = DEFAULT_SEED, device = nothing, max_steps::Int = DEFAULT_TRAIN_STEPS)
    rng = Random.Xoshiro(seed)
    alg = PPO(; n_steps = 32, batch_size = 32, epochs = 1, learning_rate = 1.0f-3)
    layer = ActorCriticLayer(
        observation_space(env),
        action_space(env);
        hidden_dims = [32, 32]
    )
    if device === nothing
        cache = init(
            RLProblem(env, layer),
            alg;
            max_steps,
            verbosity = 0,
            logger = NoTrainingLogger(),
            rng,
        )
    else
        cache = init(
            RLProblem(env, layer),
            alg;
            max_steps,
            verbosity = 0,
            logger = NoTrainingLogger(),
            rng,
            device,
        )
    end
    return cache, alg
end

function make_sac_cache(env::AbstractParallelEnv; seed::Int = DEFAULT_SEED, max_steps::Int = DEFAULT_TRAIN_STEPS)
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
    cache = init(
        RLProblem(env, layer),
        alg;
        max_steps,
        verbosity = 0,
        logger = NoTrainingLogger(),
        rng,
    )
    return cache, alg
end

function setup_rollout_collection(; n_envs::Int = DEFAULT_N_ENVS, n_steps::Int = DEFAULT_ROLLOUT_STEPS)
    env = make_cartpole_env(; n_envs = n_envs)
    cache, alg = make_ppo_cache(env; max_steps = n_steps * n_envs)
    buffer = RolloutBuffer(
        observation_space(env),
        action_space(env),
        n_steps,
        n_envs,
    )
    reset!(env)
    return env, cache, alg, buffer
end

function setup_replay_collection(; n_envs::Int = DEFAULT_N_ENVS, n_steps::Int = DEFAULT_ROLLOUT_STEPS)
    env = make_pendulum_env(; n_envs = n_envs)
    cache, alg = make_sac_cache(env; max_steps = n_steps * n_envs)
    buffer = ReplayBuffer(observation_space(env), action_space(env), 1_000)
    reset!(env)
    return env, cache, alg, buffer, n_steps
end

function setup_training_ppo(; n_envs::Int = DEFAULT_N_ENVS, max_steps::Int = DEFAULT_TRAIN_STEPS)
    env = make_cartpole_env(; n_envs = n_envs)
    cache, alg = make_ppo_cache(env; max_steps = max_steps)
    reset!(env)
    return env, cache, alg, max_steps
end

function setup_training_sac(; n_envs::Int = DEFAULT_N_ENVS, max_steps::Int = DEFAULT_TRAIN_STEPS)
    env = make_pendulum_env(; n_envs = n_envs)
    cache, alg = make_sac_cache(env; max_steps = max_steps)
    reset!(env)
    return env, cache, alg, max_steps
end

# Same workload for device benchmarks (small steps, fixed seed) so CPU vs Reactant are comparable.
const DEVICE_BENCH_MAX_STEPS = 64

function setup_training_ppo_device(; n_envs::Int = DEFAULT_N_ENVS, device = nothing)
    env = make_cartpole_env(; n_envs = n_envs)
    cache, alg = make_ppo_cache(env; device = device, max_steps = DEVICE_BENCH_MAX_STEPS)
    reset!(env)
    return env, cache, alg, DEVICE_BENCH_MAX_STEPS
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

function ppo_lux_train_state(cache)
    return Drill.lux_train_state(cache.train_state)
end

function setup_ppo_gradient_data_discrete(; n_envs::Int = DEFAULT_N_ENVS)
    env = make_cartpole_env(; n_envs = n_envs)
    cache, alg = make_ppo_cache(env)
    n_steps = alg.n_steps
    buffer = RolloutBuffer(
        observation_space(env),
        action_space(env),
        n_steps,
        n_envs,
    )
    reset!(env)
    Drill.collect_rollout!(buffer, cache, alg, env)
    Drill.prepare_rollout!(buffer, alg)
    data_loader = DataLoader(
        (
            buffer.observations,
            buffer.actions,
            buffer.advantages,
            buffer.returns,
            buffer.logprobs,
            buffer.values,
        );
        batchsize = alg.batch_size,
        shuffle = true,
        parallel = true,
        rng = cache.rng,
    )
    batch_data = nothing
    for batch_data_item in data_loader
        batch_data = batch_data_item
        break
    end
    @assert batch_data !== nothing
    train_state = deepcopy(ppo_lux_train_state(cache))
    return alg, batch_data, train_state
end

function setup_ppo_gradient_data_continuous(; n_envs::Int = DEFAULT_N_ENVS)
    env = make_pendulum_env(; n_envs = n_envs)
    cache, alg = make_ppo_cache(env)
    n_steps = alg.n_steps
    buffer = RolloutBuffer(
        observation_space(env),
        action_space(env),
        n_steps,
        n_envs,
    )
    reset!(env)
    Drill.collect_rollout!(buffer, cache, alg, env)
    Drill.prepare_rollout!(buffer, alg)
    data_loader = DataLoader(
        (
            buffer.observations,
            buffer.actions,
            buffer.advantages,
            buffer.returns,
            buffer.logprobs,
            buffer.values,
        );
        batchsize = alg.batch_size,
        shuffle = true,
        parallel = true,
        rng = cache.rng,
    )
    batch_data = nothing
    for batch_data_item in data_loader
        batch_data = batch_data_item
        break
    end
    @assert batch_data !== nothing
    train_state = deepcopy(ppo_lux_train_state(cache))
    return alg, batch_data, train_state
end

function setup_ppo_gradient_data(; n_envs::Int = DEFAULT_N_ENVS)
    return setup_ppo_gradient_data_discrete(; n_envs = n_envs)
end

function setup_sac_gradient_data(; n_envs::Int = DEFAULT_N_ENVS, n_steps::Int = DEFAULT_ROLLOUT_STEPS)
    env = make_pendulum_env(; n_envs = n_envs)
    cache, alg = make_sac_cache(env)
    n_steps = max(n_steps, cld(alg.batch_size, n_envs))
    buffer = ReplayBuffer(observation_space(env), action_space(env), alg.buffer_capacity)
    reset!(env)
    Drill.collect_rollout!(buffer, cache, alg, env, n_steps)
    data_loader = Drill.get_data_loader(buffer, alg.batch_size, 1, true, true, cache.rng)
    batch_data = nothing
    for batch_data_item in data_loader
        batch_data = batch_data_item
        break
    end
    @assert batch_data !== nothing
    return (;
        layer = cache.model,
        alg = alg,
        batch_data = batch_data,
        ts = deepcopy(cache.train_state),
        rng = cache.rng,
    )
end

function bench_sac_ad!(ad_backend, state)
    layer = state.layer
    alg = state.alg
    batch_data = state.batch_data
    ts = state.ts
    rng = state.rng
    if alg.ent_coef isa AutoEntropyCoefficient
        target_entropy = Drill.get_target_entropy(alg.ent_coef, action_space(layer))
        _, log_probs_pi, _ = Drill.action_log_prob(
            layer,
            batch_data.observations,
            Drill.parameters(ts),
            Drill.states(ts);
            rng = rng,
        )
        c = mean(log_probs_pi .+ target_entropy)
        ent_data = (; c)
        _, _, _, ent_ts = Lux.Training.compute_gradients(
            ad_backend,
            Drill.SACEntropyObjective(),
            ent_data,
            ts.ent_ts,
        )
        ts.ent_ts = ent_ts
    end
    target_q_values = Drill.compute_target_q_values(
        alg,
        layer,
        Drill.parameters(ts),
        Drill.states(ts),
        (
            rewards = batch_data.rewards,
            next_observations = batch_data.next_observations,
            terminated = batch_data.terminated,
            log_ent_coef = Drill.entropy_parameters(ts),
            target_ps = ts.target_parameters,
            target_st = ts.target_states,
        );
        rng = rng,
    )
    critic_data = (
        observations = batch_data.observations,
        actions = batch_data.actions,
        target_q_values = target_q_values,
        actor_ps = ts.actor_ts.parameters,
        actor_st = ts.actor_ts.states,
    )
    critic_objective = Drill.SACCriticObjective(alg, rng)
    _, _, _, critic_ts = Lux.Training.compute_gradients(
        ad_backend,
        critic_objective,
        critic_data,
        ts.critic_ts,
    )
    ts.critic_ts = critic_ts
    ent_coef = Float32(Drill.entropy_coefficient(ts))
    actor_objective = Drill.SACActorObjective(alg, rng)
    return Lux.Training.compute_gradients(
        ad_backend,
        actor_objective,
        (
            observations = batch_data.observations,
            ent_coef = ent_coef,
            critic_ps = ts.critic_ts.parameters,
            critic_st = ts.critic_ts.states,
        ),
        ts.actor_ts,
    )
end

end
