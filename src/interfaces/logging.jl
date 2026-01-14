# ------------------------------------------------------------
# Logging interface
# ------------------------------------------------------------
abstract type AbstractTrainingLogger end

"""
    set_step!(logger::AbstractTrainingLogger, step::Integer)

Set the global step for subsequent metric logs.
"""
function set_step! end
function increment_step! end

"""
    log_scalar!(logger::AbstractTrainingLogger, key::AbstractString, value::Real)

Log a single scalar metric under `key`.
"""
function log_scalar! end

"""
    log_metrics!(logger::AbstractTrainingLogger, kv::AbstractDict{<:AbstractString,<:Any})

Log multiple metrics at once from a string or symbol-keyed dictionary, or a NamedTuple.
If implementing a custom logger backend, only a method for the string-keyed dictionary is required to be implemented.
"""
function log_metrics! end

"""
    log_hparams!(logger::AbstractTrainingLogger, hparams::AbstractDict{<:AbstractString,<:Any}, metrics::AbstractVector{<:AbstractString})

Write hyperparameters and associate them with specified metrics for hyperparameter tuning.
"""
function log_hparams! end

"""
    flush!(logger::AbstractTrainingLogger)

Ensure any buffered data is pushed to the backend. Implementations may no-op.
"""
function flush! end

"""
    close!(logger::AbstractTrainingLogger)

Finalize the logger and release resources. Implementations may no-op.
"""
function close! end

# Strict conversion gate: identity for wrapper, error otherwise
Base.convert(::Type{AbstractTrainingLogger}, x::AbstractTrainingLogger) = x
Base.convert(::Type{AbstractTrainingLogger}, x) = error(
    "Unsupported logger $(typeof(x)). Pass NoTrainingLogger(), TensorBoardLogger.TBLogger, Wandb.WandbLogger, or a DearDiary experiment ID."
)


"""
    log_stats(env::AbstractEnv, logger::AbstractLogger) -> Nothing

Log environment-specific statistics to a logger (optional interface).

# Arguments
- `env::AbstractEnv`: The environment
- `logger::AbstractLogger`: The logger to write to

# Returns
- `Nothing`

# Notes
Default implementation does nothing. Environments can override to log custom metrics.
"""
function log_stats(env::AbstractEnv, logger::AbstractTrainingLogger)
    return nothing
end
