module DRiL_TensorBoardLoggerExt

using DRiL
using TensorBoardLogger
import DRiL: AbstractTrainingLogger, set_step!, increment_step!, log_scalar!, log_metrics!, log_hparams!, flush!, close!

mutable struct TBLoggerBackend <: AbstractTrainingLogger
    tb::TensorBoardLogger.TBLogger
    current_step::Int
end

Base.convert(::Type{AbstractTrainingLogger}, tb::TensorBoardLogger.TBLogger) = TBLoggerBackend(tb, 0)

function DRiL.set_step!(lg::TBLoggerBackend, s::Int)
    lg.current_step = s
    return TensorBoardLogger.set_step!(lg.tb, s)
end

function DRiL.increment_step!(lg::TBLoggerBackend, Δ::Int)
    lg.current_step += Δ
    TensorBoardLogger.set_step!(lg.tb, lg.current_step)
    return lg.current_step
end

function DRiL.log_scalar!(lg::TBLoggerBackend, k::AbstractString, v::Real)
    return TensorBoardLogger.log_value(lg.tb, k, v)
end

function DRiL.log_metrics!(lg::TBLoggerBackend, kv::AbstractDict{<:AbstractString, <:Any})
    for (k, v) in kv
        v isa Real && DRiL.log_scalar!(lg, k, v)
    end
    return nothing
end

function DRiL.log_hparams!(lg::TBLoggerBackend, hparams::AbstractDict{<:AbstractString, <:Any}, metrics::AbstractVector{<:AbstractString})
    # TensorBoard hparam logging using write_hparams! as per TensorBoardLogger.jl API
    hparams_dict = Dict(string(k) => v for (k, v) in hparams)
    TensorBoardLogger.write_hparams!(lg.tb, hparams_dict, String.(metrics))
    return nothing
end

DRiL.flush!(::TBLoggerBackend) = nothing
function DRiL.close!(lg::TBLoggerBackend)
    try
        close(lg.tb)
    catch
    end
    return nothing
end

end
