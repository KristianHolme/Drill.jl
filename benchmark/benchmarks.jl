using BenchmarkTools
using Drill
using ClassicControlEnvironments
using Random
using Drill.Lux
# Lux AutoZygote requires Zygote to be loaded before compute_gradients.
using Zygote

include("bench_utils.jl")
using .BenchUtils

const SUITE = BenchmarkGroup()

# Prefer a short wall-clock budget so CI stays predictable.
const BASIC_SECONDS = 1.0
const BASIC_SAMPLES = 5

rollouts = BenchmarkGroup()
SUITE["rollouts"] = rollouts

rollouts["rollout_buffer"] = @benchmarkable begin
    Drill.collect_rollout!(buffer, cache, alg, env)
end setup = begin
    env, cache, alg, buffer = BenchUtils.setup_rollout_collection()
end seconds = BASIC_SECONDS samples = BASIC_SAMPLES

rollouts["replay_buffer"] = @benchmarkable begin
    Drill.collect_rollout!(buffer, cache, alg, env, n_steps)
end setup = begin
    env, cache, alg, buffer, n_steps = BenchUtils.setup_replay_collection()
end seconds = BASIC_SECONDS samples = BASIC_SAMPLES

training = BenchmarkGroup()
SUITE["training"] = training

training["ppo_cartpole"] = @benchmarkable begin
    solve!(cache)
end setup = begin
    env, cache, alg, max_steps = BenchUtils.setup_training_ppo()
end seconds = BASIC_SECONDS samples = BASIC_SAMPLES

training["sac_pendulum"] = @benchmarkable begin
    solve!(cache)
end setup = begin
    env, cache, alg, max_steps = BenchUtils.setup_training_sac()
end seconds = BASIC_SECONDS samples = BASIC_SAMPLES

# Device benchmarks: compare CPU vs Reactant (when available). Same workload (DEVICE_BENCH_MAX_STEPS).
# Run with: BenchmarkTools.run(SUITE["devices"]). With Reactant: run Reactant entries; without, only CPU runs.
devices = BenchmarkGroup()
SUITE["devices"] = devices

devices["ppo_cpu"] = @benchmarkable begin
    solve!(cache)
end setup = begin
    env, cache, alg, max_steps = BenchUtils.setup_training_ppo_device()
end seconds = BASIC_SECONDS samples = BASIC_SAMPLES

const HAS_REACTANT = try
    @eval using Reactant
    Reactant.set_default_backend("cpu")
    true
catch
    false
end

if HAS_REACTANT && isdefined(Lux, :reactant_device)
    devices["ppo_reactant"] = @benchmarkable begin
        solve!(cache)
    end setup = begin
        env, cache, alg, max_steps = BenchUtils.setup_training_ppo_device(; device = Lux.reactant_device())
        cache.ad_type = AutoEnzyme()
        (env, cache, alg, max_steps)
    end seconds = BASIC_SECONDS samples = BASIC_SAMPLES
    # Reactant with CPU backend (no GPU): compare compiled Reactant vs plain CPU.
    devices["ppo_reactant_cpu_backend"] = @benchmarkable begin
        solve!(cache)
    end setup = begin
        if isdefined(Main, :Reactant) && isdefined(Main.Reactant, :set_default_backend)
            Main.Reactant.set_default_backend("cpu")
        end
        env, cache, alg, max_steps = BenchUtils.setup_training_ppo_device(; device = Lux.reactant_device())
        cache.ad_type = AutoEnzyme()
        (env, cache, alg, max_steps)
    end seconds = BASIC_SECONDS samples = BASIC_SAMPLES
end

wrappers = BenchmarkGroup()
SUITE["wrappers"] = wrappers

wrappers["broadcasted_act"] = @benchmarkable begin
    act!(env, actions)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end seconds = BASIC_SECONDS samples = BASIC_SAMPLES

wrappers["monitor_act"] = @benchmarkable begin
    act!(monitor_env, actions)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end seconds = BASIC_SECONDS samples = BASIC_SAMPLES

wrappers["normalize_act"] = @benchmarkable begin
    act!(normalize_env, actions)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end seconds = BASIC_SECONDS samples = BASIC_SAMPLES

wrappers["broadcasted_reset"] = @benchmarkable begin
    reset!(env)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end seconds = BASIC_SECONDS samples = BASIC_SAMPLES

wrappers["monitor_reset"] = @benchmarkable begin
    reset!(monitor_env)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end seconds = BASIC_SECONDS samples = BASIC_SAMPLES

wrappers["normalize_reset"] = @benchmarkable begin
    reset!(normalize_env)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end seconds = BASIC_SECONDS samples = BASIC_SAMPLES

wrappers["multithreaded_act"] = @benchmarkable begin
    act!(threaded_env, actions)
end setup = begin
    threaded_env, actions = BenchUtils.setup_threaded_envs()
end seconds = BASIC_SECONDS samples = BASIC_SAMPLES

wrappers["multithreaded_reset"] = @benchmarkable begin
    reset!(threaded_env)
end setup = begin
    threaded_env, actions = BenchUtils.setup_threaded_envs()
end seconds = BASIC_SECONDS samples = BASIC_SAMPLES

const HAS_ZYGOTE = try
    @eval using Zygote
    true
catch
    false
end

const HAS_ENZYME = try
    @eval using Enzyme: Reverse, set_runtime_activity
    true
catch
    false
end

const ENABLE_AD_BACKEND_BENCHES = HAS_ZYGOTE && HAS_ENZYME
if ENABLE_AD_BACKEND_BENCHES
    ad_backends = BenchmarkGroup()
    SUITE["ad_backends"] = ad_backends

    ad_backends["ppo_discrete"] = BenchmarkGroup()
    ad_backends["ppo_continuous"] = BenchmarkGroup()
    ad_backends["sac"] = BenchmarkGroup()

    ad_backend_types = [
        ("Zygote", AutoZygote()),
        ("Enzyme", AutoEnzyme()),
        ("Enzyme (with runtime activity)", AutoEnzyme(; mode = set_runtime_activity(Reverse))),
    ]

    for (name, ad_backend) in ad_backend_types
        ad_backends["ppo_discrete"][name] = @benchmarkable begin
            Lux.Training.compute_gradients($ad_backend, alg, batch_data, train_state)
        end setup = begin
            alg, batch_data, train_state = BenchUtils.setup_ppo_gradient_data_discrete()
        end seconds = BASIC_SECONDS samples = BASIC_SAMPLES
    end

    for (name, ad_backend) in ad_backend_types
        ad_backends["ppo_continuous"][name] = @benchmarkable begin
            Lux.Training.compute_gradients($ad_backend, alg, batch_data, train_state)
        end setup = begin
            alg, batch_data, train_state = BenchUtils.setup_ppo_gradient_data_continuous()
        end seconds = BASIC_SECONDS samples = BASIC_SAMPLES
    end

    for (name, ad_backend) in ad_backend_types[[1, 3]] #dont use enzyme without runtime activity
        ad_backends["sac"][name] = @benchmarkable begin
            BenchUtils.bench_sac_ad!($ad_backend, state)
        end setup = begin
            state = BenchUtils.setup_sac_gradient_data()
        end seconds = BASIC_SECONDS samples = BASIC_SAMPLES
    end
end
