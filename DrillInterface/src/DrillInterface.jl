module DrillInterface

using Random

# ------------------------------------------------------------
# Environments
# ------------------------------------------------------------

export AbstractEnv, AbstractEnvWrapper, AbstractParallelEnv, AbstractParallelEnvWrapper
export act!, observe, reset!, terminated, truncated
export action_space, get_info, number_of_envs, observation_space
export is_wrapper, unwrap, unwrap_all

"""
    AbstractEnv

Abstract base type for all reinforcement learning environments.

Subtypes must implement the following methods:
- `reset!(env)` - Reset the environment
- `act!(env, action)` - Take an action and return the reward
- `observe(env)` - Get current observation
- `terminated(env)` - Check if episode terminated
- `truncated(env)` - Check if episode was truncated
- `action_space(env)` - Get the action space
- `observation_space(env)` - Get the observation space
"""
abstract type AbstractEnv end

"""
    AbstractParallelEnv <: AbstractEnv

Abstract type for vectorized/parallel environments that manage multiple environment instances.

# Key Differences from AbstractEnv

| Method | Single Env | Parallel Env |
|--------|------------|--------------|
| `observe` | Returns one observation | Returns vector of observations |
| `act!` | Returns `reward` | Returns `(rewards, terminateds, truncateds, infos)` |
| `terminated` | Returns `Bool` | Returns `Vector{Bool}` |
| `truncated` | Returns `Bool` | Returns `Vector{Bool}` |

# Auto-Reset Behavior
Parallel environments automatically reset individual sub-environments when they terminate or truncate.
The terminal observation is stored in `infos[i]["terminal_observation"]` before reset.
"""
abstract type AbstractParallelEnv <: AbstractEnv end

"""
    reset!(env::AbstractEnv) -> Nothing

Reset the environment to its initial state.

# Arguments
- `env::AbstractEnv`: The environment to reset

# Returns
- `Nothing`
"""
function reset! end

"""
    act!(env::AbstractEnv, action) -> reward

Take an action in the environment and return the reward.

# Arguments
- `env::AbstractEnv`: The environment to act in
- `action`: The action to take (type depends on environment's action space)

# Returns
- `reward`: Numerical reward from taking the action
"""
function act! end

"""
    observe(env::AbstractEnv) -> observation

Get the current observation from the environment.

# Arguments
- `env::AbstractEnv`: The environment to observe

# Returns
- `observation`: Current state observation (type/shape depends on environment's observation space)
"""
function observe end

"""
    terminated(env::AbstractEnv) -> Bool

Check if the environment episode has terminated due to reaching a terminal state.

# Arguments
- `env::AbstractEnv`: The environment to check

# Returns
- `Bool`: `true` if episode is terminated, `false` otherwise
"""
function terminated end

"""
    truncated(env::AbstractEnv) -> Bool

Check if the environment episode has been truncated (e.g., time limit reached).

# Arguments
- `env::AbstractEnv`: The environment to check

# Returns
- `Bool`: `true` if episode is truncated, `false` otherwise
"""
function truncated end

"""
    action_space(env::AbstractEnv) -> AbstractSpace

Get the action space specification for the environment.

# Arguments
- `env::AbstractEnv`: The environment

# Returns
- `AbstractSpace`: The action space (e.g., Box, Discrete)
"""
function action_space end

"""
    observation_space(env::AbstractEnv) -> AbstractSpace

Get the observation space specification for the environment.

# Arguments
- `env::AbstractEnv`: The environment

# Returns
- `AbstractSpace`: The observation space (e.g., Box, Discrete)
"""
function observation_space end

"""
    get_info(env::AbstractEnv) -> Dict

Get additional environment information (metadata, debug info, etc.).

# Arguments
- `env::AbstractEnv`: The environment

# Returns
- `Dict`: Dictionary containing environment-specific information
"""
function get_info end

"""
    number_of_envs(env::AbstractParallelEnv) -> Int

Get the number of parallel environments in a parallel environment wrapper.

# Arguments
- `env::AbstractParallelEnv`: The parallel environment

# Returns
- `Int`: Number of parallel environments
"""
function number_of_envs end

# ------------------------------------------------------------
# Environment wrappers
# ------------------------------------------------------------
abstract type AbstractEnvWrapper{E <: AbstractEnv} <: AbstractEnv end

"""
    AbstractParallelEnvWrapper{E}

Wraps a vectorized [`AbstractParallelEnv`](@ref) (e.g. normalization or monitoring) while remaining an `AbstractParallelEnv`.
"""
abstract type AbstractParallelEnvWrapper{E <: AbstractParallelEnv} <: AbstractParallelEnv end

# ------------------------------------------------------------
# Environment wrapper utilities
# ------------------------------------------------------------
"""
    is_wrapper(env::AbstractEnv) -> Bool

Check if an environment is a wrapper around another environment.

# Arguments
- `env::AbstractEnv`: The environment to check

# Returns
- `Bool`: `true` if environment is a wrapper, `false` otherwise
"""
is_wrapper(env::AbstractEnv) = env isa AbstractEnvWrapper
is_wrapper(env::AbstractParallelEnv) = env isa AbstractParallelEnvWrapper

"""
    unwrap(env::AbstractEnvWrapper) -> AbstractEnv

Unwrap one layer of environment wrapper to access the underlying environment.

# Arguments
- `env::AbstractEnvWrapper`: The wrapped environment

# Returns
- `AbstractEnv`: The underlying environment (may still be wrapped)
"""
function unwrap end

function unwrap_all(env::AbstractEnv)
    wrapped = true
    while wrapped
        env = unwrap(env)
        wrapped = is_wrapper(env)
    end
    return env
end

function observation_space(env::AbstractParallelEnv)
    return observation_space(env.envs[1])
end

function action_space(env::AbstractParallelEnv)
    return action_space(env.envs[1])
end
# Random.seed! extensions for environments
"""
    Random.seed!(env::AbstractEnv, seed::Integer)

Seed an environment's internal RNG. Environments should have an `rng` field 
that gets seeded for reproducible behavior.
"""
function Random.seed!(env::AbstractEnv, seed::Integer)
    if hasfield(typeof(env), :rng)
        Random.seed!(env.rng, seed)
    else
        @debug "Environment $(typeof(env)) does not have an rng field - seeding has no effect"
    end
    return env
end

"""
    Random.seed!(env::AbstractParallelEnv, seed::Integer)

Seed all sub-environments in a parallel environment with incremented seeds.
Each sub-environment gets seeded with `seed + i - 1` where `i` is the environment index.
"""
function Random.seed!(env::AbstractParallelEnv, seed::Integer)
    for (i, sub_env) in enumerate(env.envs)
        Random.seed!(sub_env, seed + i - 1)
    end
    return env
end

# ------------------------------------------------------------
# Spaces
# ------------------------------------------------------------

export AbstractSpace, Box, Discrete
export batch

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

function Base.isequal(box1::Box{T1}, box2::Box{T2}) where {T1, T2}
    return T1 == T2 && box1.low == box2.low && box1.high == box2.high && box1.shape == box2.shape
end

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

"""
    rand([rng], space::Box{T}, n::Integer)

Sample `n` random values from the box space.

Returns a vector of length `n`.
"""
function Random.rand(rng::AbstractRNG, space::Box{T}, n::Integer) where {T}
    return [rand(rng, space) for _ in 1:n]
end

Random.rand(space::Box, n::Integer) = rand(Random.default_rng(), space, n)
Random.rand(space::Box) = rand(Random.default_rng(), space)

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


# ------------------------------------------------------------
# Policies
# ------------------------------------------------------------

export AbstractPolicy, RandomPolicy, ConstantPolicy

"""
    AbstractPolicy

Abstract type for policies. Subtypes are callable with signature:

    (policy::AbstractPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())

Return env-space actions for a single observation or a vector of observations.
"""
abstract type AbstractPolicy end

"""
    RandomPolicy(action_space)
    RandomPolicy(env::AbstractEnv)

A policy that returns a random action from the action space.

# Examples
```julia
using DrillInterface
space = Box(-1.0f0, 1.0f0, (2,))
policy = RandomPolicy(space)
action = policy(nothing; deterministic = true, rng = Random.Xoshiro(123))
```
"""
struct RandomPolicy{A <: AbstractSpace} <: AbstractPolicy
    action_space::A
end

function (rp::RandomPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())
    return rand(rng, rp.action_space)
end

function RandomPolicy(env::AbstractEnv)
    return RandomPolicy(action_space(env))
end


"""
    ConstantPolicy(action)

A policy that returns a constant action. Will throw an error if deterministic is false.
Will warn if rng is not nothing. Will not use the rng.

# Examples
```julia
using DrillInterface
policy = ConstantPolicy([0.0f0])
action = policy(nothing; deterministic = true)
```
"""
struct ConstantPolicy{A} <: AbstractPolicy
    action::A
end

function (cp::ConstantPolicy)(obs; deterministic::Bool = true, rng::Union{Nothing, AbstractRNG} = nothing)
    !deterministic && error("ConstantPolicy is deterministic")
    !isnothing(rng) && @warn "rng is not used by ConstantPolicy"
    return cp.action
end

# ------------------------------------------------------------
# Environment checker
# ------------------------------------------------------------
include("env_checker.jl")
export check_env

end # module
