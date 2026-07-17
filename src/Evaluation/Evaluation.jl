module Evaluation

using ProgressMeter: Progress, next!
using Random
using Statistics: mean, std

import DrillInterface: act!, number_of_envs, observation_space, observe, reset!, terminated,
    truncated, unwrap
using DrillInterface: AbstractParallelEnv, AbstractParallelEnvWrapper

import ..Wrappers: MonitorWrapperEnv
import ..Deployment: NeuralPolicy
import ..Solve: RLCache, predict_actions

include("evaluate.jl")

export evaluate, evaluate_agent

end
