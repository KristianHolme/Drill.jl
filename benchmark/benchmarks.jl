using BenchmarkTools
using Drill
using ClassicControlEnvironments
using Random
using Zygote
using Drill.Lux
using Enzyme: Reverse, set_runtime_activity
using Reactant
Reactant.set_default_backend("cpu")

include("bench_utils.jl")
using .BenchUtils

const SUITE = BenchmarkGroup()

rollouts = BenchmarkGroup()
SUITE["rollouts"] = rollouts

rollouts["rollout_buffer"] = @benchmarkable begin
    Drill.collect_rollout!(buffer, agent, alg, env)
end setup = begin
    env, agent, alg, buffer = BenchUtils.setup_rollout_collection()
end

rollouts["replay_buffer"] = @benchmarkable begin
    Drill.collect_rollout!(buffer, agent, alg, env, n_steps)
end setup = begin
    env, agent, alg, buffer, n_steps = BenchUtils.setup_replay_collection()
end

training = BenchmarkGroup()
SUITE["training"] = training

training["ppo_cartpole"] = @benchmarkable begin
    train!(agent, env, alg, max_steps)
end setup = begin
    env, agent, alg, max_steps = BenchUtils.setup_training_ppo()
end

training["sac_pendulum"] = @benchmarkable begin
    train!(agent, env, alg, max_steps)
end setup = begin
    env, agent, alg, max_steps = BenchUtils.setup_training_sac()
end

# Device benchmarks: compare CPU vs Reactant (when available). Same workload (DEVICE_BENCH_MAX_STEPS).
# Run with: BenchmarkTools.run(SUITE["devices"]). With Reactant: run Reactant entries; without, only CPU runs.
devices = BenchmarkGroup()
SUITE["devices"] = devices

devices["ppo_cpu"] = @benchmarkable begin
    train!(agent, env, alg, max_steps)
end setup = begin
    env, agent, alg, max_steps = BenchUtils.setup_training_ppo_device()
end

if isdefined(Lux, :reactant_device)
    devices["ppo_reactant"] = @benchmarkable begin
        train!(agent, env, alg, max_steps; ad_type = ad_backend)
    end setup = begin
        env, agent, alg, max_steps = BenchUtils.setup_training_ppo_device(; device = Lux.reactant_device())
        ad_backend = AutoEnzyme()
        (env, agent, alg, max_steps, ad_backend)
    end
    # Reactant with CPU backend (no GPU): compare compiled Reactant vs plain CPU.
    devices["ppo_reactant_cpu_backend"] = @benchmarkable begin
        train!(agent, env, alg, max_steps; ad_type = ad_backend)
    end setup = begin
        if isdefined(Main, :Reactant) && isdefined(Main.Reactant, :set_default_backend)
            Main.Reactant.set_default_backend("cpu")
        end
        env, agent, alg, max_steps = BenchUtils.setup_training_ppo_device(; device = Lux.reactant_device())
        ad_backend = AutoEnzyme()
        (env, agent, alg, max_steps, ad_backend)
    end
end

wrappers = BenchmarkGroup()
SUITE["wrappers"] = wrappers

wrappers["broadcasted_act"] = @benchmarkable begin
    act!(env, actions)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end

wrappers["monitor_act"] = @benchmarkable begin
    act!(monitor_env, actions)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end

wrappers["normalize_act"] = @benchmarkable begin
    act!(normalize_env, actions)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end

wrappers["broadcasted_reset"] = @benchmarkable begin
    reset!(env)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end

wrappers["monitor_reset"] = @benchmarkable begin
    reset!(monitor_env)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end

wrappers["normalize_reset"] = @benchmarkable begin
    reset!(normalize_env)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end

wrappers["multithreaded_act"] = @benchmarkable begin
    act!(threaded_env, actions)
end setup = begin
    threaded_env, actions = BenchUtils.setup_threaded_envs()
end

wrappers["multithreaded_reset"] = @benchmarkable begin
    reset!(threaded_env)
end setup = begin
    threaded_env, actions = BenchUtils.setup_threaded_envs()
end

const ENABLE_AD_BACKEND_BENCHES = true
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
        end
    end

    for (name, ad_backend) in ad_backend_types
        ad_backends["ppo_continuous"][name] = @benchmarkable begin
            Lux.Training.compute_gradients($ad_backend, alg, batch_data, train_state)
        end setup = begin
            alg, batch_data, train_state = BenchUtils.setup_ppo_gradient_data_continuous()
        end
    end

    for (name, ad_backend) in ad_backend_types[[1, 3]] #dont use enzyme without runtime activity
        # Body lives in BenchUtils so `--bench-on=main` can compare legacy and TrainState-bundle APIs.
        ad_backends["sac"][name] = @benchmarkable begin
            BenchUtils.bench_sac_ad!($ad_backend, state)
        end setup = begin
            state = BenchUtils.setup_sac_gradient_data()
        end
    end
end
