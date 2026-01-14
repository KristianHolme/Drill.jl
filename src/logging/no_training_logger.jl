struct NoTrainingLogger <: AbstractTrainingLogger end

set_step!(::NoTrainingLogger, ::Integer) = nothing
increment_step!(::NoTrainingLogger, ::Integer) = nothing
log_scalar!(::NoTrainingLogger, ::AbstractString, ::Real) = nothing
log_metrics!(::NoTrainingLogger, ::AbstractDict{<:AbstractString, <:Any}) = nothing
log_hparams!(::NoTrainingLogger, ::AbstractDict{<:AbstractString, <:Any}, ::AbstractVector{<:AbstractString}) = nothing
flush!(::NoTrainingLogger) = nothing
close!(::NoTrainingLogger) = nothing
