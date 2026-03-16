module Drill

using Accessors
using Adapt
using Base.Threads
using ChainRulesCore
using ComponentArrays
using DataStructures
import DrillInterface: act!, action_space, get_info, number_of_envs, observation_space,
    observe, reset!, terminated, truncated, unwrap, is_wrapper, unwrap_all
using DrillInterface: AbstractEnv, AbstractEnvWrapper, AbstractParallelEnv,
    AbstractParallelEnvWrapper, AbstractPolicy,
    AbstractSpace, batch, Box, Discrete, Discrete, DrillInterface
using Functors: fmap
using LinearAlgebra
using Logging
using Lux
using LoopVectorization
using MLDataDevices: AbstractDevice, cpu_device, get_device
using MLUtils
using Octavian
using Optimisers
using ProgressMeter
using Reexport
using Random
using Statistics
using StatsBase: sample
using TimerOutputs
using FileIO
using JLD2
using OneHotArrays

include("DrillDistributions/DrillDistributions.jl")
@reexport using .DrillDistributions
@reexport using DrillInterface

include("interfaces/interfaces.jl")
export AbstractAgent, AbstractAlgorithm, AbstractBuffer
export AbstractEntropyTarget, AutoEntropyTarget, FixedEntropyTarget
export AbstractEntropyCoefficient, AutoEntropyCoefficient, FixedEntropyCoefficient
export AbstractTrainingLogger, close!, flush!, increment_step!, log_hparams!, log_metrics!,
    log_scalar!, set_step!
export AbstractCallback, on_rollout_end, on_rollout_start, on_step, on_training_end,
    on_training_start
export AbstractActorCriticLayer, AbstractNoise, CriticType, QCritic, VCritic
export FeatureSharing, SeparateFeatures, SharedFeatures
export OffPolicyAlgorithm, OnPolicyAlgorithm

include("adapters/default_adapters.jl")
export AbstractActionAdapter, ClampAdapter, DiscreteAdapter, TanhScaleAdapter
export from_env, to_env

include("space_utils.jl")
export discrete_to_onehotbatch, onehotbatch_to_discrete

include("layers/layers.jl")
export ActorCriticLayer, ContinuousActorCriticLayer
export AbstractWeightInitializer, DiscreteActorCriticLayer
export OrthogonalInitializer, action_log_prob
export AbstractActorCriticLayer

include("agents/agents.jl")
export load_layer_params_and_state, predict_actions, predict_values,
    save_layer_params_and_state, steps_taken
include("agents/agent_factory.jl")
export Agent

include("buffers/buffers.jl")
export OffPolicyTrajectory, ReplayBuffer, RolloutBuffer, Trajectory

include("algorithms/traits.jl")

include("algorithms/sac.jl")
export SAC, SACLayer

include("algorithms/ppo.jl")
export PPO, load_layer_params_and_state!, train!

include("callbacks.jl")

include("environment_wrappers/scalingWrapperEnv.jl")
include("environment_wrappers/normalizeWrapperEnv.jl")
include("environment_wrappers/multithreadedParallelEnv.jl")
include("environment_wrappers/broadcastedParallelEnv.jl")
include("environment_wrappers/multiAgentParallelEnv.jl")
include("environment_wrappers/monitorWrapperEnv.jl")
export BroadcastedParallelEnv, MultiThreadedParallelEnv, NormalizeWrapperEnv, RunningMeanStd,
    ScalingWrapperEnv
export is_training, load_normalization_stats!, save_normalization_stats, set_training,
    sync_normalization_stats!
export get_original_obs, get_original_rewards, normalize_obs!, normalize_rewards!, unnormalize_obs!, unnormalize_rewards!
export EpisodeStats, MonitorWrapperEnv
export MultiAgentParallelEnv

include("deployment/deployment_policy.jl")
export NeuralPolicy, NormWrapperPolicy, extract_policy

include("utils/utils.jl")
export collect_trajectory


# New logging interface and no-op logger; concrete backends provided via package extensions
include("logging/logging_utils.jl")
include("logging/no_training_logger.jl")
export NoTrainingLogger, get_hparams

import DrillInterface: check_env

include("evaluation.jl")
export evaluate_agent

end
