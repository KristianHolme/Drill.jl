module Utils

import DrillInterface: act!, observe, reset!, terminated, truncated
using DrillInterface: Box, Discrete

import ..Wrappers: ScalingWrapperEnv, is_training, normalize_obs!, set_training,
    unscale_action!, unscale_observation!

using ComponentArrays: ComponentArray
using Functors: fmap
using OneHotArrays: onehotbatch
using Random: AbstractRNG

include("space.jl")
include("optimization.jl")
include("trajectory.jl")

export discrete_to_onehotbatch, onehotbatch_to_discrete, scale_to_space
export polyak_update!, merge_params, nested_norm, nested_scale!, nested_has_nan, nested_has_inf, nested_all_zero
export collect_trajectory

end
