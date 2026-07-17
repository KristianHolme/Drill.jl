module Solve

import CommonSolve: init, solve, solve!, step!
import Adapt: adapt, adapt_structure
using FileIO: load, save
import JLD2
import Lux
using Lux: Training
using Lux.Training: AutoZygote
import MLDataDevices: AbstractDevice, cpu_device, get_device, isleaf
using MLUtils: DataLoader
using Optimisers: adjust!
using ProgressMeter: Progress, next!
using Random: AbstractRNG, default_rng
using SciMLBase: ReturnCode
using Statistics: mean, var
using TimerOutputs: @timeit, TimerOutput, print_timer

import DrillInterface: AbstractParallelEnv, Box, Discrete, act!, action_space, batch,
    number_of_envs, observation_space, observe

import ..Adapters: AbstractActionAdapter, to_env
import ..Algorithms: AbstractAlgorithm, AutoEntropyCoefficient, OffPolicyAlgorithm,
    OnPolicyAlgorithm, PPO, SAC, PPOTrainState, SACTrainState, EntropyCoefficientLayer, action_adapter,
    compatible, compute_target_q_values, entropy_coefficient, entropy_parameters, get_gradient_steps,
    get_target_entropy, init_entropy_coefficient, lux_train_state, make_optimizer, maybe_normalize_batch_data,
    parameters, prepare_training_actions, select_actor_parameters, select_actor_states,
    select_critic_parameters, select_critic_states, set_lux_train_state!, set_states!,
    states, target_kl, SACActorObjective, SACCriticObjective, SACEntropyObjective
import ..Buffers: OffPolicyTrajectory, ReplayBuffer, RolloutBuffer, Trajectory,
    compute_gae!, get_data_loader, pack_trajectories!
import ..Callbacks: AbstractCallback, on_rollout_end, on_rollout_start, on_step,
    on_training_end, on_training_start
import ..DrillLogging: AbstractTrainingLogger, NoTrainingLogger, increment_step!,
    log_scalar!, log_stats, set_step!
import ..Layers: action_log_prob, predict_actions, predict_values, action_space as layer_action_space
import ..Problem: RLProblem, check_compatible
import ..Utils: nested_has_inf, nested_has_nan, nested_norm, nested_scale!, polyak_update!

include("runtime_cache.jl")
include("cache.jl")
include("solution.jl")
include("inference.jl")
include("collect.jl")
include("init.jl")
include("step.jl")
include("solve.jl")
include("io.jl")
include("device_adapt.jl")

export RLCache, RLSolution
export parameters, states, set_states!, invalidate_cache!, steps_taken
export get_action_and_values, predict_actions, predict_values, predict_actions_raw
export collect_trajectories, collect_rollout!, prepare_rollout!
export init, solve!, step!, solve
export save_layer_params_and_state, load_layer_params_and_state!
export current_device, canonicalize_device_batch, rollout_inference_state,
    deployment_inference_state, reactant_cache_entry_count
export rollout_action_values_kernel, rollout_predict_actions_kernel,
    rollout_predict_actions_deterministic_kernel, rollout_predict_actions_stochastic_kernel,
    rollout_predict_values_kernel
export deployment_predict_actions_kernel, deployment_predict_actions_deterministic_kernel,
    deployment_predict_actions_stochastic_kernel
export execute_rollout_action_values, execute_rollout_predict_actions,
    execute_rollout_predict_values, execute_deployment_predict_actions

end
