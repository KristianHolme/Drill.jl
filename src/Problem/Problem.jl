module Problem

import DrillInterface: action_space, observation_space

import ..Adapters: AbstractActionAdapter
import ..Algorithms: AbstractAlgorithm, action_adapter
import ..Layers: action_space as layer_action_space
import ..Layers: observation_space as layer_observation_space

include("rl_problem.jl")
include("compatibility.jl")

export RLProblem, check_compatible

end
