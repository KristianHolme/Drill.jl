# Default adapter implementations for Box and Discrete spaces

# Box spaces
function to_env(::ClampAdapter, action::AbstractArray, space::Box{T}) where {T}
    a = action
    if eltype(a) != T
        @warn "Action type mismatch: $(eltype(a)) != $T"
        a = convert.(T, a)
    end
    return clamp.(a, space.low, space.high)
end

function to_env(::TanhScaleAdapter, action::AbstractArray, space::Box{T}) where {T}
    a = action
    if eltype(a) != T
        @warn "Action type mismatch: $(eltype(a)) != $T"
        a = convert.(T, a)
    end
    # squashed with tanh, then scaled to box range
    return scale_to_space(tanh.(a), space)
end

# Optional inverse mappings (no atanh by default; map back to [-1,1] for SAC)
function from_env(::TanhScaleAdapter, action::AbstractArray, space::Box{T}) where {T}
    # Map env action in [low, high] back to [-1, 1]
    low = space.low
    high = space.high
    return T(2) .* (action .- (low .+ high) ./ T(2)) ./ (high .- low)
end

from_env(::ClampAdapter, action::AbstractArray, ::Box) = action
function onehot_to_discrete(action::OneHotVector, space::Discrete)
    return space.start + argmax(action) - 1
end
# Discrete spaces: convert onehot to discrete
function to_env(::DiscreteAdapter, action::OneHotVector, space::Discrete)
    return onehot_to_discrete(action, space)
end

from_env(::DiscreteAdapter, action, ::Discrete) = action
