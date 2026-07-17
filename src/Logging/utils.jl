function get_hparams(alg)
    @warn "get_hparams is not implemented for $(typeof(alg)). No hyperparameters will be logged."
    return Dict{String, Any}()
end

function _nt_to_string_dict(nt::NamedTuple)
    keys = propertynames(nt)
    values = getproperty.(Ref(nt), keys)
    return Dict{String, Any}(string(k) => v for (k, v) in zip(keys, values))
end

function _symbol_dict_to_string_dict(d::Dict{Symbol, Any})
    return Dict{String, Any}(string(k) => v for (k, v) in d)
end

function log_metrics!(logger::AbstractTrainingLogger, metrics::NamedTuple)
    d = _nt_to_string_dict(metrics)
    return log_metrics!(logger, d)
end

function log_metrics!(logger::AbstractTrainingLogger, metrics::Dict{Symbol, Any})
    d = _symbol_dict_to_string_dict(metrics)
    return log_metrics!(logger, d)
end

# Algorithm-specific get_hparams methods (PPO, SAC, etc.) will be added when Algorithms is wired.
