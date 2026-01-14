module DRiL_WandbExt

using DRiL
using Wandb
import DRiL: AbstractTrainingLogger, set_step!, increment_step!, log_scalar!, log_metrics!, log_hparams!, flush!, close!

mutable struct WandbBackend <: AbstractTrainingLogger
    wb::Wandb.WandbLogger
    current_step::Int
end

Base.convert(::Type{AbstractTrainingLogger}, wb::Wandb.WandbLogger) = WandbBackend(wb, 0)

# Only track step internally - don't use Wandb.increment_step! since we pass step= explicitly to log()
DRiL.set_step!(lg::WandbBackend, s::Integer) = (lg.current_step = s)
DRiL.increment_step!(lg::WandbBackend, Δ::Integer) = (lg.current_step += Δ; lg.current_step)

function DRiL.log_scalar!(lg::WandbBackend, k::AbstractString, v::Real)
    Wandb.log(lg.wb, Dict(k => v); step = lg.current_step)
    return nothing
end

function DRiL.log_metrics!(lg::WandbBackend, kv::AbstractDict{<:AbstractString, <:Any})
    d = Dict{String, Any}()
    for (k, v) in kv
        d[string(k)] = v
    end
    Wandb.log(lg.wb, d; step = lg.current_step)
    return nothing
end

function DRiL.log_hparams!(lg::WandbBackend, hparams::AbstractDict{<:AbstractString, <:Any}, metrics::AbstractVector{<:AbstractString})
    # Wandb hyperparameter logging using update_config! as per Wandb.jl API
    config_dict = Dict{String, Any}()
    for (k, v) in hparams
        config_dict[string(k)] = v
    end
    # Wandb doesn't require explicit metric association like TensorBoard
    Wandb.update_config!(lg.wb, config_dict)
    return nothing
end

DRiL.flush!(::WandbBackend) = nothing
function DRiL.close!(lg::WandbBackend)
    try
        close(lg.wb)
    catch
    end
    return nothing
end

end
