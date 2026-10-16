module Buffers

import DataStructures: capacity, isfull
using DrillInterface: AbstractSpace, Box
import DrillInterface: action_space, observation_space, reset!
using MLUtils: DataLoader
using Random: AbstractRNG

include("types.jl")
include("rollout.jl")
include("replay.jl")

export AbstractBuffer, OnPolicyBuffer, OffPolicyBuffer
export RolloutBuffer, ReplayBuffer
export compute_gae!, get_data_loader, step_indices, store_step!
export add_transitions!, sample_batch

end
