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

function to_env(::ScaleAdapter, action::AbstractArray, space::Box{T}) where {T}
    a = action
    if eltype(a) != T
        @warn "Action type mismatch: $(eltype(a)) != $T"
        a = convert.(T, a)
    end
    # The policy already squashes with tanh, so `a` lies in [-1, 1]; scale it to the box.
    low = space.low
    high = space.high
    return a .* (high - low) ./ T(2) + (low + high) ./ T(2)
end

# Inverse of `to_env`: map an env action in [low, high] back to [-1, 1]
function from_env(::ScaleAdapter, action::AbstractArray, space::Box{T}) where {T}
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
function to_env(::DiscreteAdapter, action::AbstractVector, space::Discrete)
    return space.start + argmax(action) - 1
end
function to_env(::DiscreteAdapter, action::Integer, space::Discrete)
    @assert action in space "Action $(action) is out of bounds for $(space)"
    return action
end

from_env(::DiscreteAdapter, action, ::Discrete) = action
