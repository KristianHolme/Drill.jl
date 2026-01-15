module DRiL

using Accessors
using Base.Threads
using ChainRulesCore
using ComponentArrays
using DataStructures
using LinearAlgebra
using Logging
using Lux
using LoopVectorization
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

include("DRiLDistributions/DRiLDistributions.jl")
@reexport using .DRiLDistributions

include("interfaces/interfaces.jl")
export AbstractEnv, AbstractParallelEnv, AbstractAgent, AbstractBuffer, AbstractAlgorithm
export AbstractEntropyTarget, FixedEntropyTarget, AutoEntropyTarget
export AbstractEntropyCoefficient, FixedEntropyCoefficient, AutoEntropyCoefficient
export reset!, act!, observe, terminated, truncated, action_space, observation_space, get_info, number_of_envs
export AbstractTrainingLogger, set_step!, increment_step!, log_scalar!, log_metrics!, flush!, close!, log_hparams!
export AbstractCallback, on_training_start, on_training_end, on_rollout_start, on_rollout_end, on_step
export AbstractActorCriticLayer, AbstractNoise, CriticType, QCritic, VCritic
export FeatureSharing, SharedFeatures, SeparateFeatures
export OffPolicyAlgorithm, OnPolicyAlgorithm
export AbstractPolicy

include("spaces.jl")
export AbstractSpace, Box, Discrete

include("adapters/default_adapters.jl")
export AbstractActionAdapter, ClampAdapter, TanhScaleAdapter, DiscreteAdapter
export to_env, from_env

include("layers/layers.jl")
export ActorCriticLayer, ContinuousActorCriticLayer
export DiscreteActorCriticLayer, AbstractWeightInitializer
export OrthogonalInitializer, action_log_prob
export AbstractActorCriticLayer

include("agents/agents.jl")
export predict_actions, predict_values, steps_taken, save_policy_params_and_state, load_policy_params_and_state
include("agents/agent_factory.jl")
export Agent

include("buffers/buffers.jl")
export Trajectory, RolloutBuffer, OffPolicyTrajectory, ReplayBuffer

include("algorithms/traits.jl")

include("algorithms/sac.jl")
export SAC, SACLayer

include("algorithms/ppo.jl")
export train!, PPO, load_policy_params_and_state!

include("callbacks.jl")

include("environment_wrappers/scalingWrapperEnv.jl")
include("environment_wrappers/normalizeWrapperEnv.jl")
include("environment_wrappers/multithreadedParallelEnv.jl")
include("environment_wrappers/broadcastedParallelEnv.jl")
include("environment_wrappers/multiAgentParallelEnv.jl")
include("environment_wrappers/monitorWrapperEnv.jl")
include("environment_wrappers/wrapper_utils.jl")
export MultiThreadedParallelEnv, BroadcastedParallelEnv, ScalingWrapperEnv, NormalizeWrapperEnv, RunningMeanStd
export save_normalization_stats, load_normalization_stats!, set_training, is_training
export get_original_obs, get_original_rewards, normalize_obs!, normalize_rewards!, unnormalize_obs!, unnormalize_rewards!
export MonitorWrapperEnv, EpisodeStats, is_wrapper, unwrap, unwrap_all
export MultiAgentParallelEnv

include("deployment/deployment_policy.jl")
export extract_policy, DeploymentPolicy, NormalizedDeploymentPolicy, RandomPolicy, ZeroPolicy

include("utils/utils.jl")
export collect_trajectory


# New logging interface and no-op logger; concrete backends provided via package extensions
include("logging/logging_utils.jl")
include("logging/no_training_logger.jl")
export get_hparams, NoTrainingLogger

include("env_checker.jl")
export check_env

include("evaluation.jl")
export evaluate_agent

end
