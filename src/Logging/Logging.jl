module DrillLogging

import DrillInterface: AbstractEnv

include("types.jl")
include("no_training_logger.jl")
include("utils.jl")

export AbstractTrainingLogger
export set_step!, increment_step!, log_scalar!, log_metrics!, log_hparams!, flush!, close!
export log_stats
export NoTrainingLogger, get_hparams

end
