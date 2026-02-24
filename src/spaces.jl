"""
    AbstractSpace

Abstract base type for all observation and action spaces in Drill.jl.
Concrete subtypes include `Box` (continuous) and `Discrete` (finite actions).
"""
abstract type AbstractSpace end

"""
    Box{T <: Number} <: AbstractSpace

A continuous space with lower and upper bounds per dimension.

# Fields
- `low::Array{T}`: Lower bounds for each dimension
- `high::Array{T}`: Upper bounds for each dimension
- `shape::Tuple{Vararg{Int}}`: Shape of the space

# Example
```julia
# 2D box with different bounds per dimension
space = Box(Float32[-1, -2], Float32[1, 3])

# Uniform bounds
space = Box(-1.0f0, 1.0f0, (4,))
```
"""
struct Box{T <: Number} <: AbstractSpace
    low::Array{T}
    high::Array{T}
    shape::Tuple{Vararg{Int}}
end
function Box{T}(low::Array{T}, high::Array{T}) where {T <: Number}
    @assert size(low) == size(high) "Low and high arrays must have the same shape"
    @assert all(low .<= high) "All low values must be <= corresponding high values"
    shape = size(low)
    return Box{T}(low, high, shape)
end
function Box(low::T, high::T, shape::Tuple{Vararg{Int}}) where {T <: Number}
    return Box{T}(low * ones(T, shape), high * ones(T, shape), shape)
end

# Convenience constructors
Box(low::Array{T}, high::Array{T}) where {T <: Number} = Box{T}(low, high)

Base.ndims(space::Box) = length(size(space))

Base.eltype(::Box{T}) where {T} = T

#TODO fix comparison of spaces
function Base.isequal(box1::Box{T1}, box2::Box{T2}) where {T1, T2}
    return T1 == T2 && box1.low == box2.low && box1.high == box2.high && box1.shape == box2.shape
end

# Extend Random.rand for Box spaces
"""
    rand([rng], space::Box{T})

Sample a random value from the box space with potentially different bounds per dimension.

# Examples
```julia
low = Float32[-1.0, -2.0]
high = Float32[1.0, 3.0]
space = Box(low, high)
sample = rand(space)
# Returns a 2-element Float32 array with values in [-1,1] and [-2,3] respectively
```
"""
function Random.rand(rng::AbstractRNG, space::Box{T}) where {T}
    # Generate random values in [0, 1] with correct type and shape
    unit_random = rand(rng, T, space.shape...)
    # Scale to [low, high] range element-wise
    return unit_random .* (space.high .- space.low) .+ space.low
end

# Multiple samples version
"""
    rand([rng], space::Box{T}, n::Integer)

Sample `n` random values from the box space.

Returns a vector of length `n`.
"""
function Random.rand(rng::AbstractRNG, space::Box{T}, n::Integer) where {T}
    return [rand(rng, space) for _ in 1:n]
end

Random.rand(space::Box, n::Integer) = rand(Random.default_rng(), space, n)

"""
    sample in space::Box{T}

Check if a sample is within the bounds of the box space.

# Examples
```julia
low = Float32[-1.0, -2.0]
high = Float32[1.0, 3.0]
space = Box(low, high)
Float32[0.5, 1.5] in space  # Returns true
Float32[1.5, 0.0] in space  # Returns false (first element out of bounds)

# Can also use ∈ symbol
@test action ∈ action_space
```
"""
function Base.in(sample, space::Box{T}) where {T}
    if !isa(sample, AbstractArray)
        return false
    end

    # Check shape compatibility (allowing for batch dimensions)
    sample_shape = size(sample)
    if length(sample_shape) < length(space.shape)
        return false
    end

    # Check if the leading dimensions match the space shape
    if sample_shape[1:length(space.shape)] != space.shape
        return false
    end

    # Check type compatibility - require exact type match for strict type safety
    if eltype(sample) != T
        return false
    end

    # Check bounds element-wise
    return all(space.low .<= sample .<= space.high)
end


function scale_to_space(action, action_space::Box{T}) where {T}
    low = action_space.low
    high = action_space.high
    x = action
    return x .* (high - low) ./ T(2) + (low + high) ./ T(2)
end


"""
    Discrete{T <: Integer} <: AbstractSpace

A discrete space representing a finite set of integer actions.

# Fields
- `n::T`: Number of discrete actions
- `start::T`: Lowest action value

# Example
```julia
space = Discrete(4)     # Actions: 1, 2, 3, 4
space = Discrete(4, 0)  # Actions: 0, 1, 2, 3
```
"""
struct Discrete{T <: Integer} <: AbstractSpace
    n::T
    start::T
    function Discrete(n::T, start::T = 1) where {T <: Integer}
        @assert n > 0 "n must be positive"
        return new{T}(n, start)
    end
end

Base.ndims(::Discrete) = 1

Base.eltype(::Discrete{T}) where {T <: Integer} = T

function Base.isequal(disc1::Discrete, disc2::Discrete)
    return disc1.n == disc2.n && disc1.start == disc2.start
end

"""
    rand([rng], space::Discrete)

Sample an integer action in `space.start:(space.start + space.n - 1)`.
"""
function Random.rand(rng::AbstractRNG, space::Discrete)
    return rand(rng, space.start:(space.start + space.n - 1))
end

Random.rand(space::Discrete) = rand(Random.default_rng(), space)

"""
    rand([rng], space::Discrete, n::Integer)

Sample `n` integer actions from the discrete space.
"""
function Random.rand(rng::AbstractRNG, space::Discrete, n::Integer)
    return rand(rng, space.start:(space.start + space.n - 1), n)
end

Random.rand(space::Discrete, n::Integer) = rand(Random.default_rng(), space, n)

"""
    sample in space::Discrete

Check if an integer sample is within the discrete action range.
"""
function Base.in(sample::Integer, space::Discrete)
    return space.start <= sample <= (space.start + space.n - 1)
end

Base.in(sample, space::Discrete) = false #non integers are not in space

function process_action(
        action::Integer,
        action_space::Discrete,
        alg::AbstractAlgorithm,
    )
    @assert action in action_space "Action $(action) is out of bounds for Discrete($(action_space.n), $(action_space.start))"
    return action
end

function process_action(
        actions::AbstractVector{<:Integer},
        action_space::Discrete,
        alg::AbstractAlgorithm,
    )
    return process_action.(actions, Ref(action_space), Ref(alg))
end

function process_action(
        actions::AbstractMatrix{<:Integer},
        action_space::Discrete,
        alg::AbstractAlgorithm,
    )
    return process_action.(actions, Ref(action_space), Ref(alg))
end

Base.size(::Discrete) = (1,)
Base.size(space::Box) = space.shape

"""
    batch(x::AbstractArray, space::AbstractSpace)

Batch an array of observations or actions.
"""
function batch end

batch(x::AbstractArray, space::Box) = stack(x)

function batch(x::AbstractVector{<:Integer}, space::Discrete)
    x_int = convert(Vector{eltype(space)}, collect(x))
    return reshape(x_int, 1, :)
end

function batch(x::AbstractMatrix{<:Integer}, space::Discrete)
    return x
end

function discrete_to_onehotbatch(actions::AbstractArray{<:Integer}, space::Discrete)
    flat_actions = vec(actions)
    indices = map(flat_actions) do action
        idx = action - space.start + 1
        @assert 1 <= idx <= space.n "Action $(action) is out of bounds for Discrete($(space.n), $(space.start))"
        return idx
    end
    return OneHotArrays.onehotbatch(indices, 1:space.n)
end

function onehotbatch_to_discrete(actions::AbstractMatrix, space::Discrete)
    idx = argmax(actions; dims = 1)
    return [space.start + idx[i][1] - 1 for i in eachindex(idx)]
end
