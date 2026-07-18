module Solve

import CommonSolve: init, solve, solve!, step!
import Adapt: adapt, adapt_structure
using FileIO: load, save
import JLD2
import Lux
using Lux: Training
using Lux.Training: AutoZygote
import MLDataDevices: AbstractDevice, cpu_device, get_device, isleaf
import ProgressMeter
using ProgressMeter: Progress, next!
using Random: AbstractRNG, default_rng
using SciMLBase: ReturnCode
using TimerOutputs: TimerOutput, print_timer

import DrillInterface: AbstractParallelEnv, Box, Discrete, act!, action_space, batch,
    number_of_envs, observation_space, observe

import ..Adapters: AbstractActionAdapter, to_env
import ..Algorithms: AbstractAlgorithm, OffPolicyAlgorithm, OnPolicyAlgorithm,
    PPO, SAC, PPOTrainState, SACTrainState, EntropyCoefficientLayer, action_adapter,
    compatible, init_entropy_coefficient, make_optimizer, parameters,
    select_actor_parameters, select_actor_states, select_critic_parameters,
    select_critic_states, set_states!, states, train_step!
import ..Buffers: OffPolicyTrajectory, ReplayBuffer, RolloutBuffer, Trajectory,
    compute_gae!, pack_trajectories!
const _Drill = parentmodule(@__MODULE__)
const AbstractCallback = _Drill.AbstractCallback
const on_rollout_end = _Drill.on_rollout_end
const on_rollout_start = _Drill.on_rollout_start
const on_step = _Drill.on_step
const on_training_end = _Drill.on_training_end
const on_training_start = _Drill.on_training_start
const RLProblem = _Drill.RLProblem
const check_compatible = _Drill.check_compatible
import ..DrillLogging: AbstractTrainingLogger, NoTrainingLogger
import ..Models: predict_actions, predict_values, action_space as model_action_space

include("verbosity.jl")
include("runtime_cache.jl")
include("cache.jl")
include("progress.jl")
include("solution.jl")
include("inference.jl")
include("collect.jl")
include("init.jl")
include("step.jl")
include("solve.jl")
include("io.jl")
include("device_adapt.jl")

export RLCache, RLSolution
export Verbosity, DEFAULT_VERBOSITY, normalize_verbosity, print_training_table
export parameters, states, set_states!, invalidate_cache!, steps_taken
export get_action_and_values, predict_actions, predict_values, predict_actions_raw
export collect_trajectories, collect_rollout!, prepare_rollout!
export init, solve!, step!, solve
export save_model_params_and_state, load_model_params_and_state!
export current_device, canonicalize_device_batch, rollout_inference_state,
    deployment_inference_state, reactant_cache_entry_count
export get_device, cpu_device
export rollout_action_values_kernel, rollout_predict_actions_kernel,
    rollout_predict_actions_deterministic_kernel, rollout_predict_actions_stochastic_kernel,
    rollout_predict_values_kernel
export deployment_predict_actions_kernel, deployment_predict_actions_deterministic_kernel,
    deployment_predict_actions_stochastic_kernel
export execute_rollout_action_values, execute_rollout_predict_actions,
    execute_rollout_predict_values, execute_deployment_predict_actions

# Used by Algorithms train_step! implementations (late-included).
export _record_stat!, _mark_complete!, _callbacks_continue, add_steps!, add_gradient_update!
export update_training_progress!, latest_stat, training_metric_rows

end
