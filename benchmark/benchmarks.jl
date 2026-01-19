using BenchmarkTools
using DRiL
using ClassicControlEnvironments
using Random
using Zygote

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
