struct RLProblem{E, M, U, A}
    env::E
    model::M
    u0::U
    adapter::A
end

function RLProblem(env, model; u0 = nothing, adapter = nothing)
    return RLProblem{typeof(env), typeof(model), typeof(u0), typeof(adapter)}(
        env,
        model,
        u0,
        adapter,
    )
end

function _spaces_compatible(a, b)
    return typeof(a) == typeof(b) && size(a) == size(b) && eltype(a) == eltype(b)
end

function check_compatible(prob::RLProblem, alg::AbstractAlgorithm)
    env_obs_space = observation_space(prob.env)
    env_action_space = action_space(prob.env)
    model_obs_space = observation_space(prob.model)
    model_action_space = action_space(prob.model)

    _spaces_compatible(env_obs_space, model_obs_space) ||
        throw(ArgumentError("Environment and model observation spaces are incompatible."))
    _spaces_compatible(env_action_space, model_action_space) ||
        throw(ArgumentError("Environment and model action spaces are incompatible."))

    if prob.adapter === nothing
        action_adapter(alg, env_action_space)
    elseif !(prob.adapter isa AbstractActionAdapter)
        throw(ArgumentError("adapter must be nothing or an AbstractActionAdapter."))
    end
    return true
end
