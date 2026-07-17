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

function _init_train_state(model, alg::PPO, ps, st)
    optimizer = make_optimizer(alg)
    return PPOTrainState(Training.TrainState(model, ps, st, optimizer))
end

function _init_train_state(model, alg::SAC, ps, st, device)
    actor_ps = select_actor_parameters(model, ps)
    critic_ps = select_critic_parameters(model, ps)
    actor_st = select_actor_states(model, st)
    critic_st = select_critic_states(model, st)
    actor_ts = Training.TrainState(model, actor_ps, actor_st, make_optimizer(alg))
    critic_ts = Training.TrainState(model, critic_ps, critic_st, make_optimizer(alg))
    ent_ts = Training.TrainState(
        EntropyCoefficientLayer(),
        device(init_entropy_coefficient(alg.ent_coef)),
        NamedTuple(),
        make_optimizer(alg),
    )
    return SACTrainState(actor_ts, critic_ts, ent_ts, deepcopy(critic_ps), deepcopy(critic_st))
end

function init(
        prob::RLProblem,
        alg::AbstractAlgorithm;
        max_steps::Int,
        callback = nothing,
        callbacks = nothing,
        logger = NoTrainingLogger(),
        verbosity::Int = 1,
        verbose::Union{Nothing, Int} = nothing,
        rng::AbstractRNG = default_rng(),
        buffer = nothing,
        ad_type::Training.AbstractADType = AutoZygote(),
        device = cpu_device(),
    )
    check_compatible(prob, alg)
    ps, st = _initial_ps_st(prob, rng)
    ps = device(ps)
    st = device(st)
    train_state = if alg isa SAC
        _init_train_state(prob.model, alg, ps, st, device)
    else
        _init_train_state(prob.model, alg, ps, st)
    end
    selected_buffer = buffer === nothing ? _default_buffer(prob, alg) : buffer
    if buffer !== nothing && !compatible(alg, selected_buffer)
        throw(ArgumentError("Buffer $(typeof(selected_buffer)) is incompatible with algorithm $(typeof(alg))."))
    end
    selected_adapter = prob.adapter === nothing ? action_adapter(alg, action_space(prob.env)) : prob.adapter
    selected_callbacks = _normalize_callbacks(; callback, callbacks)
    selected_logger = convert(AbstractTrainingLogger, logger)
    cache = RLCache(
        prob,
        alg,
        prob.model,
        selected_adapter,
        train_state,
        selected_buffer,
        selected_logger,
        rng,
        verbose === nothing ? verbosity : verbose,
        selected_callbacks,
        max_steps,
        0,
        0,
        ad_type,
        ReturnCode.Default,
        Dict{Symbol, Any}(),
        TimerOutput(),
        nothing,
        alg.optimizer,
        Dict{Symbol, Any}(),
    )
    if alg isa SAC
        n_envs = number_of_envs(prob.env)
        total_start_steps = alg.start_steps > 0 ? alg.start_steps : alg.train_freq * n_envs
        adjusted_total_start_steps = max(1, div(total_start_steps, n_envs)) * n_envs
        cache.workspace[:next_collect_steps] = div(adjusted_total_start_steps, n_envs)
        cache.workspace[:sac_iteration] = 0
    end
    return cache
end
