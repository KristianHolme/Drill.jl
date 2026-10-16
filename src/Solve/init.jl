function _normalize_callbacks(; callback = nothing, callbacks = nothing)
    selected = callbacks === nothing ? callback : callbacks
    if selected === nothing
        return AbstractCallback[]
    elseif selected isa AbstractCallback
        return AbstractCallback[selected]
    end
    return collect(selected)
end

function _initial_ps_st(prob::RLProblem, rng::AbstractRNG)
    if prob.u0 === nothing
        return Lux.setup(rng, prob.model)
    elseif prob.u0 isa NamedTuple && haskey(prob.u0, :ps) && haskey(prob.u0, :st)
        return prob.u0.ps, prob.u0.st
    elseif prob.u0 isa Tuple && length(prob.u0) == 2
        return prob.u0
    end
    throw(ArgumentError("u0 must be nothing, a NamedTuple with ps/st, or a two-tuple."))
end

function _default_buffer(prob::RLProblem, alg::PPO)
    obs_space = observation_space(prob.env)
    T = eltype(obs_space)
    dtype = T <: AbstractFloat ? T : Float32
    return RolloutBuffer(
        obs_space,
        action_space(prob.env),
        alg.n_steps,
        number_of_envs(prob.env);
        dtype,
    )
end

function _default_buffer(prob::RLProblem, alg::SAC)
    return ReplayBuffer(observation_space(prob.env), action_space(prob.env), alg.buffer_capacity)
end

function init_workspace!(cache::RLCache, ::AbstractAlgorithm)
    return nothing
end

function init_workspace!(cache::RLCache, alg::SAC)
    n_envs = number_of_envs(cache.prob.env)
    total_start_steps = alg.start_steps > 0 ? alg.start_steps : alg.train_freq * n_envs
    adjusted_total_start_steps = max(1, div(total_start_steps, n_envs)) * n_envs
    cache.workspace[:next_collect_steps] = div(adjusted_total_start_steps, n_envs)
    cache.workspace[:sac_iteration] = 0
    return nothing
end

function init(
        prob::RLProblem,
        alg::AbstractAlgorithm;
        max_steps::Int,
        callback = nothing,
        callbacks = nothing,
        logger = NoTrainingLogger(),
        verbosity = DEFAULT_VERBOSITY,
        rng::AbstractRNG = default_rng(),
        buffer = nothing,
        ad_type::AbstractADType = AutoZygote(),
        device = cpu_device(),
    )
    check_compatible(prob, alg)
    ps, st = _initial_ps_st(prob, rng)
    learner = init_learner(alg, prob.model, device(ps), device(st); rng = device_rng(device, rng), device)
    selected_buffer = buffer === nothing ? _default_buffer(prob, alg) : buffer
    if buffer !== nothing && !compatible(alg, selected_buffer)
        throw(ArgumentError("Buffer $(typeof(selected_buffer)) is incompatible with algorithm $(typeof(alg))."))
    end
    selected_adapter = prob.adapter === nothing ? action_adapter(alg, action_space(prob.env)) : prob.adapter
    selected_callbacks = _normalize_callbacks(; callback, callbacks)
    selected_logger = convert(AbstractTrainingLogger, logger)
    selected_verbosity = normalize_verbosity(verbosity)
    progress_meter = Progress(
        max_steps;
        desc = "Training...",
        showspeed = true,
        enabled = selected_verbosity.meter > 0,
    )
    cache = RLCache(
        prob,
        alg,
        prob.model,
        selected_adapter,
        learner,
        selected_buffer,
        selected_logger,
        rng,
        selected_verbosity,
        progress_meter,
        selected_callbacks,
        max_steps,
        0,
        0,
        ad_type,
        ReturnCode.Default,
        Dict{Symbol, Any}(),
        selected_verbosity.timer == 0 ? NoTimerOutput() : TimerOutput(),
        nothing,
        Dict{Symbol, Any}(),
    )
    init_workspace!(cache, alg)
    return cache
end
