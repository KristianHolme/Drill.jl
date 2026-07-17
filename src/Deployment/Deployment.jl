module Deployment

using Adapt
using MLDataDevices
using Random

import DrillInterface: AbstractPolicy, batch, observation_space
using DrillInterface: AbstractEnv

import ..Adapters: AbstractActionAdapter, to_env
import ..Layers: predict_actions
import ..Layers: action_space as layer_action_space
import ..Layers: observation_space as layer_observation_space
import ..Wrappers: NormalizeWrapperEnv, RunningMeanStd, normalize_obs!
import ..Solve: RLCache, canonicalize_device_batch, current_device,
    deployment_inference_state, execute_deployment_predict_actions, parameters, states

include("policy.jl")

export NeuralPolicy, NormWrapperPolicy, extract_policy

end
