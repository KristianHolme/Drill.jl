module DRiL_DearDiaryExt

using DRiL
using DearDiary
import DRiL: AbstractTrainingLogger, set_step!, increment_step!, log_scalar!, log_dict!, log_hparams!, flush!, close!

"""
    DearDiaryBackend <: AbstractTrainingLogger

A logging backend that uses DearDiary.jl for experiment tracking.

DearDiary uses an iteration-based model:
- Each training run corresponds to an Experiment
- Each logging step corresponds to an Iteration
- Metrics and parameters are stored per iteration

The backend lazily creates iterations when metrics are logged, and caches the current
iteration to avoid creating multiple iterations for the same step.
"""
mutable struct DearDiaryBackend <: AbstractTrainingLogger
    experiment_id::Int64
    current_step::Int
    current_iteration_id::Union{Int64, Nothing}
    hparams_logged::Bool
end

"""
    DearDiaryBackend(experiment_id::Integer)

Create a DearDiary logging backend for the given experiment.

# Arguments
- `experiment_id::Integer`: The ID of the DearDiary experiment to log to.

# Example
```julia
using DearDiary
using DRiL

# Initialize database and create project/experiment
DearDiary.initialize_database()
project_id, _ = create_project("My RL Project")
experiment_id, _ = create_experiment(project_id, DearDiary.IN_PROGRESS, "PPO Training")

# Create the backend
backend = DRiL.DearDiaryBackend(experiment_id)
```
"""
function DearDiaryBackend(experiment_id::Integer)
    return DearDiaryBackend(Int64(experiment_id), 0, nothing, false)
end

# Convert DearDiary experiment ID directly to backend
Base.convert(::Type{AbstractTrainingLogger}, experiment_id::Integer) = DearDiaryBackend(experiment_id)

"""
    ensure_iteration!(lg::DearDiaryBackend)

Ensure an iteration exists for the current step. Creates one if needed.
Returns the iteration ID.
"""
function ensure_iteration!(lg::DearDiaryBackend)
    if isnothing(lg.current_iteration_id)
        result = DearDiary.create_iteration(lg.experiment_id)
        if isnothing(result.id)
            @warn "Failed to create DearDiary iteration for step $(lg.current_step)"
            return nothing
        end
        lg.current_iteration_id = result.id
    end
    return lg.current_iteration_id
end

function DRiL.set_step!(lg::DearDiaryBackend, s::Integer)
    if s != lg.current_step
        lg.current_step = s
        # Invalidate current iteration - will create new one on next log
        lg.current_iteration_id = nothing
    end
    return nothing
end

function DRiL.increment_step!(lg::DearDiaryBackend, Δ::Integer)
    lg.current_step += Δ
    # Invalidate current iteration - will create new one on next log
    lg.current_iteration_id = nothing
    return lg.current_step
end

function DRiL.log_scalar!(lg::DearDiaryBackend, k::AbstractString, v::Real)
    iteration_id = ensure_iteration!(lg)
    if !isnothing(iteration_id)
        DearDiary.create_metric(iteration_id, string(k), Float64(v))
    end
    return nothing
end

function DRiL.log_dict!(lg::DearDiaryBackend, kv::AbstractDict{<:AbstractString, <:Any})
    iteration_id = ensure_iteration!(lg)
    if isnothing(iteration_id)
        return nothing
    end
    for (k, v) in kv
        if v isa Real
            DearDiary.create_metric(iteration_id, string(k), Float64(v))
        end
    end
    return nothing
end

function DRiL.log_hparams!(lg::DearDiaryBackend, hparams::AbstractDict{<:AbstractString, <:Any}, metrics::AbstractVector{<:AbstractString})
    # DearDiary stores parameters per iteration, so we create an iteration for hparams
    # and log all hyperparameters there. Only do this once per experiment.
    if lg.hparams_logged
        return nothing
    end

    iteration_id = ensure_iteration!(lg)
    if isnothing(iteration_id)
        return nothing
    end

    for (k, v) in hparams
        # DearDiary parameters accept string or Real values
        if v isa Real
            DearDiary.create_parameter(iteration_id, string(k), v)
        else
            DearDiary.create_parameter(iteration_id, string(k), string(v))
        end
    end
    lg.hparams_logged = true
    return nothing
end

# DearDiary uses SQLite which handles buffering internally
DRiL.flush!(::DearDiaryBackend) = nothing

function DRiL.close!(lg::DearDiaryBackend)
    # Mark experiment as completed if desired
    # For now, just reset the backend state
    lg.current_iteration_id = nothing
    return nothing
end

end
