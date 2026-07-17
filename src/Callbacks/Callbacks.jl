module Callbacks

include("types.jl")
include("hooks.jl")

export AbstractCallback
export on_training_start, on_rollout_start, on_step, on_rollout_end, on_training_end

end
