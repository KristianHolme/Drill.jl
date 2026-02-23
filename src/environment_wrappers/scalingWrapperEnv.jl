struct ScalingWrapperEnv{E <: AbstractEnv, O <: AbstractSpace, A <: AbstractSpace} <: AbstractEnvWrapper{E}
    env::E
    observation_space::O
    action_space::A
    orig_observation_space::O
    orig_action_space::A
    # Pre-computed scaling factors for observations
    obs_scale_factor::Array{eltype(O)}
    obs_offset::Array{eltype(O)}
    # Pre-computed scaling factors for actions
    act_scale_factor::Array{eltype(A)}
    act_offset::Array{eltype(A)}
end

function ScalingWrapperEnv(env::E) where {E <: AbstractEnv}
    orig_obs_space = observation_space(env)
    orig_act_space = action_space(env)

    return ScalingWrapperEnv(env, orig_obs_space, orig_act_space)
end

function ScalingWrapperEnv(env::E, original_obs_space::Box, original_act_space::Box) where {E <: AbstractEnv}
    # Create new observation space with bounds [-1, 1]
    T_obs = eltype(original_obs_space)
    T_act = eltype(original_act_space)

    scaled_obs_space = @set original_obs_space.low = -1 * ones(T_obs, size(original_obs_space.low))
    scaled_obs_space = @set scaled_obs_space.high = 1 * ones(T_obs, size(original_obs_space.high))

    # Create new action space with bounds [-1, 1]
    scaled_act_space = @set original_act_space.low = -1 * ones(T_act, size(original_act_space.low))
    scaled_act_space = @set scaled_act_space.high = 1 * ones(T_act, size(original_act_space.high))

    # Pre-compute scaling factors for observations: scale = 2 / (high - low), offset = low
    obs_range = original_obs_space.high .- original_obs_space.low
    obs_scale_factor = 2 ./ obs_range
    obs_offset = original_obs_space.low

    # Pre-compute scaling factors for actions
    act_range = original_act_space.high .- original_act_space.low
    act_scale_factor = 2 ./ act_range
    act_offset = original_act_space.low

    return ScalingWrapperEnv{E, Box, Box}(
        env, scaled_obs_space, scaled_act_space, original_obs_space, original_act_space,
        obs_scale_factor, obs_offset, act_scale_factor, act_offset
    )
end
#TODO:document/fix unwrap
Drill.unwrap(env::ScalingWrapperEnv) = env.env

function observation_space(env::ScalingWrapperEnv)
    return env.observation_space
end

function action_space(env::ScalingWrapperEnv)
    return env.action_space
end

function reset!(env::ScalingWrapperEnv)
    reset!(env.env)
    return nothing
end

# Fast in-place scaling functions using pre-computed factors
# PERFORMANCE NOTES:
# - Use the in-place versions (scale_observation!, unscale_action!, etc.) when possible
# - Pre-allocate output buffers and reuse them across calls
# - Scaling factors are pre-computed once during construction to avoid repeated calculations
# - All operations use @. macro for vectorized, allocation-free broadcasting
@inline function scale!(input, scale_factor, offset)
    @. input = (input - offset) * scale_factor - 1
    return nothing
end

@inline function unscale!(input, scale_factor, offset)
    @. input = (input + 1) / scale_factor + offset
    return nothing
end

# Allocating versions for compatibility
function scale_observation!(observation, env::ScalingWrapperEnv{E, Box, Box}) where {E}
    scale!(observation, env.obs_scale_factor, env.obs_offset)
    return nothing
end

function unscale_observation!(observation, env::ScalingWrapperEnv{E, Box, Box}) where {E}
    unscale!(observation, env.obs_scale_factor, env.obs_offset)
    return nothing
end


function observe(env::ScalingWrapperEnv{E, Box, Box}) where {E}
    orig_obs = observe(env.env)
    # Scale observation from original space to [-1, 1] using pre-computed factors
    scale_observation!(orig_obs, env)
    return orig_obs
end


function scale_action!(action, env::ScalingWrapperEnv{E, Box, Box}) where {E}
    scale!(action, env.act_scale_factor, env.act_offset)
    return nothing
end

function unscale_action!(action, env::ScalingWrapperEnv{E, Box, Box}) where {E}
    unscale!(action, env.act_scale_factor, env.act_offset)
    return nothing
end

function act!(env::ScalingWrapperEnv{E, Box, Box}, action) where {E}
    unscale_action!(action, env)
    return act!(env.env, action)
end

function terminated(env::ScalingWrapperEnv)
    return terminated(env.env)
end

function truncated(env::ScalingWrapperEnv)
    return truncated(env.env)
end

function get_info(env::ScalingWrapperEnv)
    return get_info(env.env)
end


"""
    Random.seed!(env::ScalingWrapperEnv, seed::Integer)

Seed a wrapped environment by forwarding the seed to the underlying environment.
"""
function Random.seed!(env::ScalingWrapperEnv, seed::Integer)
    Random.seed!(env.env, seed)
    return env
end


# ==============================================================================
# Show methods for nice environment display
# ==============================================================================


# ScalingWrapperEnv show methods
function Base.show(io::IO, env::ScalingWrapperEnv{E, O, A}) where {E, O, A}
    if get(io, :compact, false)
        return print(io, "ScalingWrapperEnv{", E, "}")
    end
    println(io, "ScalingWrapperEnv{", E, "}")
    println(io, "  - Scaled observation bounds: [-1, 1]")
    println(io, "  - Scaled action bounds: [-1, 1]")
    println(io, "  - Original observation space: ", env.orig_observation_space)
    println(io, "  - Original action space: ", env.orig_action_space)
    print(io, "  wrapped environment: ")
    return show(io, env.env)
end
