using BenchmarkTools
using DRiL
using ClassicControlEnvironments
using Random
using Zygote
import Lux
using Lux: AutoZygote, AutoEnzyme
using Enzyme: Reverse, set_runtime_activity

include("bench_utils.jl")
using .BenchUtils

const SUITE = BenchmarkGroup()

rollouts = BenchmarkGroup()
SUITE["rollouts"] = rollouts

rollouts["rollout_buffer"] = @benchmarkable begin
    DRiL.collect_rollout!(buffer, agent, alg, env)
end setup = begin
    env, agent, alg, buffer = BenchUtils.setup_rollout_collection()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

rollouts["replay_buffer"] = @benchmarkable begin
    DRiL.collect_rollout!(buffer, agent, alg, env, n_steps)
end setup = begin
    env, agent, alg, buffer, n_steps = BenchUtils.setup_replay_collection()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

training = BenchmarkGroup()
SUITE["training"] = training

training["ppo_cartpole"] = @benchmarkable begin
    train!(agent, env, alg, max_steps)
end setup = begin
    env, agent, alg, max_steps = BenchUtils.setup_training_ppo()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

training["sac_pendulum"] = @benchmarkable begin
    train!(agent, env, alg, max_steps)
end setup = begin
    env, agent, alg, max_steps = BenchUtils.setup_training_sac()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

wrappers = BenchmarkGroup()
SUITE["wrappers"] = wrappers

wrappers["broadcasted_act"] = @benchmarkable begin
    act!(env, actions)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

wrappers["monitor_act"] = @benchmarkable begin
    act!(monitor_env, actions)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

wrappers["normalize_act"] = @benchmarkable begin
    act!(normalize_env, actions)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

wrappers["broadcasted_reset"] = @benchmarkable begin
    reset!(env)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

wrappers["monitor_reset"] = @benchmarkable begin
    reset!(monitor_env)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

wrappers["normalize_reset"] = @benchmarkable begin
    reset!(normalize_env)
end setup = begin
    env, monitor_env, normalize_env, actions = BenchUtils.setup_wrapper_envs()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

wrappers["multithreaded_act"] = @benchmarkable begin
    act!(threaded_env, actions)
end setup = begin
    threaded_env, actions = BenchUtils.setup_threaded_envs()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

wrappers["multithreaded_reset"] = @benchmarkable begin
    reset!(threaded_env)
end setup = begin
    threaded_env, actions = BenchUtils.setup_threaded_envs()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

const ENABLE_AD_BACKEND_BENCHES = false
if ENABLE_AD_BACKEND_BENCHES
    ad_backends = BenchmarkGroup()
    SUITE["ad_backends"] = ad_backends

    ad_backends["ppo"] = BenchmarkGroup()
    ad_backends["sac"] = BenchmarkGroup()

    ad_backend_types = [
        ("Zygote", AutoZygote()),
        ("Enzyme", AutoEnzyme()),
        ("Enzyme (with runtime activity)", AutoEnzyme(; mode = set_runtime_activity(Reverse))),
    ]

    for (name, ad_backend) in ad_backend_types
        ad_backends["ppo"][name] = @benchmarkable begin
            Lux.Training.compute_gradients(ad_backend, alg, batch_data, train_state)
        end setup = begin
            alg, batch_data, train_state = BenchUtils.setup_ppo_gradient_data()
        end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES
    end

    for (name, ad_backend) in ad_backend_types
        ad_backends["sac"][name] = @benchmarkable begin
            if alg.ent_coef isa AutoEntropyCoefficient
                target_entropy = DRiL.get_target_entropy(alg.ent_coef, action_space(layer))
                ent_data = (
                    observations = batch_data.observations,
                    layer_ps = train_state.parameters,
                    layer_st = train_state.states,
                    target_entropy = target_entropy,
                    target_ps = target_ps,
                    target_st = target_st,
                )
                _, _, _, ent_train_state = Lux.Training.compute_gradients(
                    ad_backend,
                    (model, ps, st, data) -> DRiL.sac_ent_coef_loss(alg, layer, ps, st, data; rng = rng),
                    ent_data,
                    ent_train_state,
                )
            end
            critic_data = (
                observations = batch_data.observations,
                actions = batch_data.actions,
                rewards = batch_data.rewards,
                terminated = batch_data.terminated,
                truncated = batch_data.truncated,
                next_observations = batch_data.next_observations,
                log_ent_coef = ent_train_state.parameters,
                target_ps = target_ps,
                target_st = target_st,
            )
            _, _, _, train_state = Lux.Training.compute_gradients(
                ad_backend,
                (model, ps, st, data) -> DRiL.sac_critic_loss(alg, layer, ps, st, data; rng = rng),
                critic_data,
                train_state,
            )
            Lux.Training.compute_gradients(
                ad_backend,
                (model, ps, st, data) -> DRiL.sac_actor_loss(alg, layer, ps, st, data; rng = rng),
                (
                    observations = batch_data.observations,
                    actions = batch_data.actions,
                    rewards = batch_data.rewards,
                    terminated = batch_data.terminated,
                    truncated = batch_data.truncated,
                    next_observations = batch_data.next_observations,
                    log_ent_coef = ent_train_state.parameters,
                ),
                train_state,
            )
        end setup = begin
            layer, alg, batch_data, train_state, ent_train_state, target_ps, target_st, rng =
                BenchUtils.setup_sac_gradient_data()
        end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES
    end
end
