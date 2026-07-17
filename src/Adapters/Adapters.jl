module Adapters

import DrillInterface: Box, Discrete
using OneHotArrays: OneHotVector

include("types.jl")
include("default_adapters.jl")

export AbstractActionAdapter, ClampAdapter, TanhScaleAdapter, DiscreteAdapter
export to_env, from_env

end
