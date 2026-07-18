module Drill

import Adapt
import DataStructures
import DrillInterface
import Lux
import MLDataDevices
import ProgressMeter
using ProgressMeter: Progress, next!
using Random: AbstractRNG, default_rng
using Reexport: @reexport
using Statistics: mean, std

import DrillInterface: AbstractParallelEnv, AbstractParallelEnvWrapper, AbstractPolicy,
    act!, action_space, batch, number_of_envs, observation_space, observe, reset!, unwrap
import MLDataDevices: AbstractDevice, cpu_device

include("DrillDistributions/DrillDistributions.jl")
@reexport using .DrillDistributions
@reexport using DrillInterface

include("Logging/Logging.jl")
@reexport using .DrillLogging

include("Wrappers/Wrappers.jl")
@reexport using .Wrappers

include("Utils/Utils.jl")
@reexport using .Utils

include("Adapters/Adapters.jl")
@reexport using .Adapters

include("Models/Models.jl")
@reexport using .Models

include("Buffers/Buffers.jl")
@reexport using .Buffers

include("callbacks.jl")

include("Algorithms/Algorithms.jl")
@reexport using .Algorithms

include("problem.jl")

include("Solve/Solve.jl")
@reexport using .Solve

# Algorithm train steps need RLCache / collect APIs from Solve.
Base.include(Algorithms, joinpath(@__DIR__, "Algorithms", "ppo_step.jl"))
Base.include(Algorithms, joinpath(@__DIR__, "Algorithms", "sac_step.jl"))

include("deployment.jl")
include("evaluate.jl")

export AbstractCallback, on_training_start, on_rollout_start, on_step, on_rollout_end, on_training_end
export RLProblem, check_compatible
export NeuralPolicy, NormWrapperPolicy, extract_policy
export evaluate

end
