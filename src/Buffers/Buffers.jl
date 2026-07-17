module Buffers

using DataStructures: CircularBuffer
import DataStructures: capacity, isfull
using DrillInterface: AbstractSpace, Box
import DrillInterface: action_space, observation_space, reset!
using DrillInterface: batch
using MLUtils: DataLoader
using Random: AbstractRNG
using StatsBase: sample

include("types.jl")
include("trajectory.jl")
include("rollout.jl")
include("replay.jl")

export AbstractBuffer
export BufferKind, OnPolicyBuffer, OffPolicyBuffer, buffer_kind
export RolloutBuffer, Trajectory, OffPolicyTrajectory, ReplayBuffer
export compute_advantages!, compute_gae!, get_data_loader, pack_trajectories!

end
