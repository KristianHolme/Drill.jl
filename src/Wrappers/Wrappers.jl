module Wrappers

using Base.Threads: @threads
using Accessors: @set
using DataStructures: CircularBuffer
using FileIO: save, load
import JLD2
import Random: seed!
using Statistics: mean, var

import DrillInterface
import DrillInterface: act!, action_space, get_info, number_of_envs, observation_space,
    observe, reset!, terminated, truncated, unwrap
using DrillInterface: AbstractEnv, AbstractEnvWrapper, AbstractParallelEnv,
    AbstractParallelEnvWrapper, AbstractSpace, Box, batch

import ..DrillLogging: AbstractTrainingLogger, log_scalar!

include("scaling.jl")
include("normalize.jl")
include("broadcasted_parallel.jl")
include("multithreaded_parallel.jl")
include("multi_agent_parallel.jl")
include("monitor.jl")

export BroadcastedParallelEnv, MultiThreadedParallelEnv, MultiAgentParallelEnv
export ScalingWrapperEnv, NormalizeWrapperEnv, MonitorWrapperEnv
export RunningMeanStd, EpisodeStats
export update!, update_from_moments!
export is_training, set_training
export get_original_obs, get_original_rewards
export normalize_obs!, normalize_rewards!, unnormalize_obs!, unnormalize_rewards!
export save_normalization_stats, load_normalization_stats!, sync_normalization_stats!

end
