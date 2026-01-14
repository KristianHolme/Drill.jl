function get_hparams(alg::AbstractAlgorithm)
    @warn "get_hparams is not implemented for $(typeof(alg)). No hyperparameters will be logged."
    return Dict{String, Any}()
end

function get_hparams(agent::AbstractAgent)
    @warn "get_hparams is not implemented for $(typeof(agent)). No hyperparameters will be logged."
    return Dict{String, Any}()
end

function get_hparams(alg::PPO)
    hparams = Dict{String, Any}(
        "gamma" => alg.gamma,
        "gae_lambda" => alg.gae_lambda,
        "clip_range" => alg.clip_range,
        "ent_coef" => alg.ent_coef,
        "vf_coef" => alg.vf_coef,
        "max_grad_norm" => alg.max_grad_norm,
        "normalize_advantage" => alg.normalize_advantage,
        "learning_rate" => alg.learning_rate,
        "batch_size" => alg.batch_size,
        "n_steps" => alg.n_steps,
        "epochs" => alg.epochs
    )

    if !isnothing(alg.clip_range_vf)
        hparams["clip_range_vf"] = alg.clip_range_vf
    end

    if !isnothing(alg.target_kl)
        hparams["target_kl"] = alg.target_kl
    end

    return hparams
end

function get_hparams(alg::SAC)
    hparams = Dict{String, Any}(
        "learning_rate" => alg.learning_rate,
        "buffer_capacity" => alg.buffer_capacity,
        "start_steps" => alg.start_steps,
        "batch_size" => alg.batch_size,
        "tau" => alg.tau,
        "gamma" => alg.gamma,
        "train_freq" => alg.train_freq,
        "gradient_steps" => alg.gradient_steps,
        "target_update_interval" => alg.target_update_interval
    )

    hparams = merge(hparams, get_hparams(alg.ent_coef))
    return hparams
end

function get_hparams(ent_coef::AutoEntropyCoefficient)
    return Dict{String, Any}(
        "ent_coef_mode" => "auto",
        "ent_coef_value" => ent_coef.initial_value
    )
end

function get_hparams(ent_coef::FixedEntropyCoefficient)
    return Dict{String, Any}(
        "ent_coef_mode" => "fixed",
        "ent_coef_value" => ent_coef.coef
    )
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
