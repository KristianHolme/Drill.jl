"""
Environment checking utilities for DrillInterface.

Verify that an environment implements the interface and respects its spaces.
"""

"""
    check_env(env; warn::Bool = true, verbose::Bool = true) -> true

Check that a single ([`AbstractEnv`](@ref)) or parallel ([`AbstractParallelEnv`](@ref))
environment follows the DrillInterface and respects its observation space.

Throws an `AssertionError` on an interface violation. Non-critical issues are reported
as warnings when `warn` is `true`.
"""
function check_env(env::Union{AbstractEnv, AbstractParallelEnv}; warn::Bool = true, verbose::Bool = true)
    verbose && @info "Checking environment implementation..."
    _check_required_methods(env; verbose)

    obs_space = observation_space(env)
    act_space = action_space(env)
    _check_space_properties(obs_space, "observation_space"; verbose)
    _check_space_properties(act_space, "action_space"; verbose)

    _check_reset(env, obs_space; warn, verbose)
    _check_step(env, obs_space, act_space; warn, verbose)
    _check_space_constraints(env, obs_space, act_space; warn, verbose)

    verbose && @info "✅ Environment check completed successfully!"
    return true
end

function _required_methods(::AbstractEnv)
    return [
        (reset!, "reset!(env; seed)"),
        (observe, "observe(env)"),
        (terminated, "terminated(env)"),
        (truncated, "truncated(env)"),
        (action_space, "action_space(env)"),
        (observation_space, "observation_space(env)"),
        (act!, "act!(env, action)"),
    ]
end

function _required_methods(::AbstractParallelEnv)
    return [
        (reset!, "reset!(penv; seed)"),
        (observe, "observe(penv)"),
        (step!, "step!(penv, actions)"),
        (number_of_envs, "number_of_envs(penv)"),
        (action_space, "action_space(penv)"),
        (observation_space, "observation_space(penv)"),
    ]
end

function _check_required_methods(env; verbose::Bool = true)
    verbose && @info "Checking required method implementations..."
    for (f, name) in _required_methods(env)
        if isempty(methods(f, (typeof(env),))) && isempty(methods(f, (typeof(env), Any)))
            throw(AssertionError("Environment must implement method: $name"))
        end
        verbose && @info "  ✓ $name implemented"
    end
    return nothing
end

function _check_space_properties(space::Box, space_name::String; verbose::Bool = true)
    @assert all(space.low .<= space.high) "$space_name: low bounds must be <= high bounds"
    @assert length(space.shape) > 0 "$space_name: shape must be non-empty"
    verbose && @info "  ✓ $(space_name): $(space.shape) $(eltype(space)) [$(space.low), $(space.high)]"
    return nothing
end

function _check_space_properties(space::Discrete, space_name::String; verbose::Bool = true)
    verbose && @info "  ✓ $(space_name): Discrete($(space.n), start = $(space.start))"
    return nothing
end

function _check_space_properties(space::AbstractSpace, space_name::String; verbose::Bool = true)
    @warn "Space type $(typeof(space)) not fully supported in checker"
    verbose && @info "  ⚠ $(space_name): $(typeof(space)) (limited checks)"
    return nothing
end

function _check_reset(env::AbstractEnv, obs_space; warn::Bool = true, verbose::Bool = true)
    verbose && @info "Checking reset functionality..."
    @assert isnothing(reset!(env; seed = 42)) "reset!(env; seed) must return nothing"
    @assert terminated(env) isa Bool "terminated(env) must return Bool"
    @assert truncated(env) isa Bool "truncated(env) must return Bool"
    @assert !terminated(env) "Environment should not be terminated immediately after reset"
    @assert !truncated(env) "Environment should not be truncated immediately after reset"
    obs1 = copy(observe(env))
    _check_observation_shape(obs1, obs_space, "observe(env) after reset!")

    reset!(env; seed = 42)
    obs2 = observe(env)
    if warn && obs1 != obs2
        @warn "reset!(env; seed = 42) twice gave different observations; seeding may not be implemented"
    end

    if obs2 isa AbstractArray
        dest = similar(obs2)
        @assert observe!(dest, env) === dest "observe!(dest, env) must return dest"
        _check_observation_shape(dest, obs_space, "observe!(dest, env)")
    end
    verbose && @info "  ✓ reset!, observe and observe! work"
    return nothing
end

function _check_reset(penv::AbstractParallelEnv, obs_space; warn::Bool = true, verbose::Bool = true)
    verbose && @info "Checking reset functionality..."
    n = number_of_envs(penv)
    obs1 = copy(reset!(penv; seed = 42))
    _check_batched_observation(obs1, obs_space, n, "reset!(penv)")
    _check_batched_observation(observe(penv), obs_space, n, "observe(penv)")
    obs2 = reset!(penv; seed = 42)
    if warn && obs1 != obs2
        @warn "reset!(penv; seed = 42) twice gave different observations; seeding may not be implemented"
    end
    verbose && @info "  ✓ reset! and observe work"
    return nothing
end

function _check_step(env::AbstractEnv, obs_space, act_space; warn::Bool = true, verbose::Bool = true)
    verbose && @info "Checking step functionality..."
    reset!(env)
    action = rand(act_space)
    action_before = copy(action)
    reward = act!(env, action)
    @assert reward isa Real "act!(env, action) must return a real reward"
    @assert action == action_before "act! must not mutate the action it is given"
    @assert terminated(env) isa Bool "terminated(env) must return Bool"
    @assert truncated(env) isa Bool "truncated(env) must return Bool"
    _check_observation_shape(observe(env), obs_space, "observe(env) after act!")
    verbose && @info "  ✓ act! works"
    return nothing
end

function _check_step(penv::AbstractParallelEnv, obs_space, act_space; warn::Bool = true, verbose::Bool = true)
    verbose && @info "Checking step functionality..."
    n = number_of_envs(penv)
    reset!(penv)
    actions = rand(act_space, n)
    result = step!(penv, actions)
    @assert result isa Tuple && length(result) == 6 "step!(penv, actions) must return (obs, rewards, terminated, truncated, final_obs, infos)"
    obs, rewards, terms, truncs, final_obs, infos = result
    _check_batched_observation(obs, obs_space, n, "step! obs")
    _check_batched_observation(final_obs, obs_space, n, "step! final_obs")
    @assert length(rewards) == n "rewards length must match n_envs"
    @assert length(terms) == n && eltype(terms) == Bool "terminated must be a Bool vector of length n_envs"
    @assert length(truncs) == n && eltype(truncs) == Bool "truncated must be a Bool vector of length n_envs"
    @assert isnothing(infos) || length(infos) == n "infos must be nothing or have length n_envs"
    verbose && @info "  ✓ step! works"
    return nothing
end

function _check_space_constraints(env::AbstractEnv, obs_space, act_space; warn::Bool = true, verbose::Bool = true)
    verbose && @info "Checking space constraints..."
    violations = 0
    for _ in 1:3
        reset!(env)
        for _ in 1:20
            observe(env) in obs_space || (violations += 1)
            act!(env, rand(act_space))
            (terminated(env) || truncated(env)) && break
        end
    end
    _report_violations(violations; warn, verbose)
    return nothing
end

function _check_space_constraints(penv::AbstractParallelEnv, obs_space, act_space; warn::Bool = true, verbose::Bool = true)
    verbose && @info "Checking space constraints..."
    n = number_of_envs(penv)
    violations = 0
    obs = reset!(penv)
    for _ in 1:20
        for i in 1:n
            collect(observation_slot(obs, i)) in obs_space || (violations += 1)
        end
        obs, _ = step!(penv, rand(act_space, n))
    end
    _report_violations(violations; warn, verbose)
    return nothing
end

function _report_violations(violations::Int; warn::Bool, verbose::Bool)
    if violations > 0
        warn && @warn "Found $violations observation space constraint violations"
    else
        verbose && @info "  ✓ Environment respects space constraints"
    end
    return nothing
end

function _check_observation_shape(obs, obs_space::Box, context::String)
    @assert obs isa AbstractArray "$context: observation must be an AbstractArray for a Box space"
    @assert size(obs) == obs_space.shape "$context: observation shape $(size(obs)) != expected $(obs_space.shape)"
    @assert eltype(obs) == eltype(obs_space) "$context: observation type $(eltype(obs)) != expected $(eltype(obs_space))"
    return nothing
end

function _check_observation_shape(obs, obs_space::Discrete, context::String)
    @assert obs in obs_space "$context: observation $obs is not in $obs_space"
    return nothing
end

function _check_observation_shape(obs, obs_space::AbstractSpace, context::String)
    throw(AssertionError("$context: Unsupported observation space type: $(typeof(obs_space))"))
end

function _check_batched_observation(obs, obs_space::AbstractSpace, n::Int, context::String)
    @assert obs isa AbstractArray "$context: batched observation must be an AbstractArray"
    expected = (size(obs_space)..., n)
    @assert size(obs) == expected "$context: batched observation shape $(size(obs)) != expected $expected"
    @assert eltype(obs) == eltype(obs_space) "$context: observation type $(eltype(obs)) != expected $(eltype(obs_space))"
    return nothing
end
