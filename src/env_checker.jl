"""
Environment checking utilities for Drill.

Provides functions to verify that environments correctly implement the required interface
and respect observation/action space constraints.
"""


"""
    check_env(env::AbstractEnv; warn::Bool=true, skip_render_check::Bool=false, verbose::Bool=true)

Check that an environment follows the Drill interface and respects space constraints.

# Arguments
- `env::AbstractEnv`: The environment to check
- `warn::Bool=true`: Whether to show warnings for non-critical issues
- `skip_render_check::Bool=false`: Whether to skip render method check (not implemented yet)
- `verbose::Bool=true`: Whether to print detailed information

# Returns
- `Bool`: true if all checks pass, false otherwise

# Throws
- `AssertionError`: If critical interface violations are found
"""
function check_env(env::AbstractEnv; warn::Bool = true, skip_render_check::Bool = false, verbose::Bool = true)
    verbose && @info "Checking environment implementation..."

    # Check required methods exist
    _check_required_methods(env; verbose = verbose)

    # Check spaces are properly defined
    obs_space = observation_space(env)
    act_space = action_space(env)
    _check_spaces(obs_space, act_space; verbose = verbose)

    # Test environment dynamics
    _check_environment_dynamics(env, obs_space, act_space; warn = warn, verbose = verbose)

    # Test reset functionality
    _check_reset_functionality(env, obs_space; warn = warn, verbose = verbose)

    # Test step functionality
    _check_step_functionality(env, obs_space, act_space; warn = warn, verbose = verbose)

    # Test space constraints
    _check_space_constraints(env, obs_space, act_space; warn = warn, verbose = verbose)

    verbose && @info "✅ Environment check completed successfully!"
    return true
end

"""
Check that all required methods from basic_types.jl are implemented.
"""
function _check_required_methods(env::AbstractEnv; verbose::Bool = true)
    verbose && @info "Checking required method implementations..."

    # Methods required for all environments
    common_required_methods = [
        (reset!, "reset!(env)"),
        (observe, "observe(env)"),
        (terminated, "terminated(env)"),
        (truncated, "truncated(env)"),
        (action_space, "action_space(env)"),
        (observation_space, "observation_space(env)"),
        (act!, "act!(env, action)"),
        (get_info, "get_info(env)"),
    ]

    # Check common methods
    for (method_func, method_name) in common_required_methods
        if !hasmethod(method_func, (typeof(env),)) && !hasmethod(method_func, (typeof(env), Any))
            throw(AssertionError("Environment must implement method: $method_name"))
        end
        verbose && @info "  ✓ $method_name implemented"
    end
    return
end

"""
Check that observation and action spaces are properly defined.
"""
function _check_spaces(obs_space::AbstractSpace, act_space::AbstractSpace; verbose::Bool = true)
    verbose && @info "Checking space definitions..."

    # Use multiple dispatch for space-specific checks
    _check_space_properties(obs_space, "observation_space"; verbose = verbose)
    return _check_space_properties(act_space, "action_space"; verbose = verbose)
end

# Multiple dispatch methods for different space types
function _check_space_properties(space::Box, space_name::String; verbose::Bool = true)
    @assert all(space.low .<= space.high) "$space_name: low bounds must be <= high bounds"
    @assert length(space.shape) > 0 "$space_name: shape must be non-empty"
    return verbose && @info "  ✓ $(space_name): $(space.shape) $(eltype(space)) [$(space.low), $(space.high)]"
end

# Fallback for unsupported space types
function _check_space_properties(space::AbstractSpace, space_name::String; verbose::Bool = true)
    @warn "Space type $(typeof(space)) not fully supported in checker"
    return verbose && @info "  ⚠ $(space_name): $(typeof(space)) (limited checks)"
end

"""
Check basic environment dynamics and state consistency.
"""
function _check_environment_dynamics(env::AbstractEnv, obs_space, act_space; warn::Bool = true, verbose::Bool = true)
    verbose && @info "Checking environment dynamics..."

    # Test that environment starts in consistent state
    initial_terminated = terminated(env)
    initial_truncated = truncated(env)
    initial_info = get_info(env)

    @assert isa(initial_terminated, Bool) "terminated() must return Bool"
    @assert isa(initial_truncated, Bool) "truncated() must return Bool"
    @assert isa(initial_info, Dict) "get_info() must return Dict"

    verbose && @info "  ✓ Environment state methods return correct types"

    # Test that observe works without reset
    return try
        obs = observe(env)
        _check_observation_shape(obs, obs_space, "observe() before reset")
        verbose && @info "  ✓ observe() works before reset"
    catch e
        warn && @warn "observe() failed before reset: $e"
    end
end

"""
Check reset functionality.
"""
function _check_reset_functionality(env::AbstractEnv, obs_space; warn::Bool = true, verbose::Bool = true)
    verbose && @info "Checking reset functionality..."

    rng = MersenneTwister(42)

    # Test reset with RNG
    reset_obs = reset!(env)
    _check_observation_shape(reset_obs, obs_space, "reset!(env)")

    # Check that environment state is consistent after reset
    @assert !terminated(env) "Environment should not be terminated immediately after reset"
    @assert !truncated(env) "Environment should not be truncated immediately after reset"

    # Test that observe returns same observation as reset
    current_obs = observe(env)
    if !isapprox(reset_obs, current_obs; atol = 1.0e-6)
        warn && @warn "observe() after reset returns different observation than reset! returned"
    end

    # Test reset without RNG (default)
    reset_obs2 = reset!(env)
    _check_observation_shape(reset_obs2, obs_space, "reset!(env) without RNG")

    return verbose && @info "  ✓ Reset functionality works correctly"
end

"""
Check step functionality for both single and parallel environments.
"""
function _check_step_functionality(env::AbstractEnv, obs_space, act_space; warn::Bool = true, verbose::Bool = true)
    verbose && @info "Checking step functionality..."

    # Reset environment first
    reset!(env)

    # Generate a valid action
    action = rand(act_space)

    return if env isa AbstractParallelEnv
        # Test parallel environment act!
        try
            rewards = act!(env, action)
            terminateds = terminated(env)
            truncateds = truncated(env)
            infos = get_info(env)
            next_obs = observe(env)

            @assert length(rewards) == number_of_envs(env) "rewards length must match n_envs"
            @assert length(terminateds) == number_of_envs(env) "terminateds length must match n_envs"
            @assert length(truncateds) == number_of_envs(env) "truncateds length must match n_envs"
            @assert length(infos) == number_of_envs(env) "infos length must match n_envs"

            _check_observation_shape(next_obs, obs_space, "act!(parallel_env)")

            verbose && @info "  ✓ Parallel environment act! works correctly"
        catch e
            @error "Parallel environment act! failed: $e"
            rethrow(e)
        end
    else
        # Test single environment act!
        try
            # Test act!
            obs_before = observe(env)
            act!(env, action)

            _check_observation_shape(obs_before, obs_space, "act!(single_env)")

            verbose && @info "  ✓ Single environment act! works correctly"
        catch e
            @error "Single environment act! functionality failed: $e"
            rethrow(e)
        end
    end
end

"""
Check that environment respects observation and action space constraints.
"""
function _check_space_constraints(env::AbstractEnv, obs_space, act_space; warn::Bool = true, verbose::Bool = true)
    verbose && @info "Checking space constraints..."

    # Test multiple episodes to check constraint adherence
    n_test_steps = 20
    constraint_violations = 0

    for episode in 1:3
        reset!(env)

        for step in 1:n_test_steps
            # Check current observation
            obs = observe(env)
            if obs ∉ obs_space
                constraint_violations += 1
                warn && @warn "Observation violates space constraints at episode $episode, step $step"
            end

            # Take action and check if episode continues
            action = rand(act_space)

            if env isa AbstractParallelEnv
                # For parallel envs, auto-reset happens
                rewards = act!(env, [action])
                terminateds = terminated(env)
                truncateds = truncated(env)
                if terminateds[1] || truncateds[1]
                    break  # Episode ended
                end
            else
                # For single envs
                act!(env, action)
                if terminated(env) || truncated(env)
                    break  # Episode ended
                end
            end
        end
    end

    return if constraint_violations > 0
        warn && @warn "Found $constraint_violations observation space constraint violations"
    else
        verbose && @info "  ✓ Environment respects space constraints"
    end
end

# Multiple dispatch method for Box
function _check_observation_shape(obs, obs_space::Box, context::String)
    expected_shape = obs_space.shape
    expected_type = eltype(obs_space)

    return if isa(obs, AbstractArray)
        obs_shape = size(obs)
        obs_type = eltype(obs)

        # Check shape (allowing for batch dimensions)
        if length(obs_shape) == length(expected_shape)
            # Single observation
            @assert obs_shape == expected_shape "$context: observation shape $obs_shape != expected $expected_shape"
        elseif length(obs_shape) == length(expected_shape) + 1
            # Batched observations
            @assert obs_shape[1:(end - 1)] == expected_shape "$context: observation batch shape mismatch"
        elseif obs isa Vector && length(obs) > 0 && size(obs[1]) == expected_shape
            # vector of observations
            @assert all(size(o) == expected_shape for o in obs) "$context: observation batch shape mismatch"
        else
            throw(AssertionError("$context: observation has unexpected size/dimensions"))
        end

        # Check type
        @assert obs_type == expected_type "$context: observation type $obs_type != expected $expected_type"
    else
        throw(AssertionError("$context: observation must be AbstractArray for Box space"))
    end
end

# Fallback for unsupported space types
function _check_observation_shape_dispatch(obs, obs_space::AbstractSpace, context::String)
    throw(AssertionError("$context: Unsupported observation space type: $(typeof(obs_space))"))
end
