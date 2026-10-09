module DrillInterface

using Random: Random, AbstractRNG
import CommonSolve: step!

# ------------------------------------------------------------
# Environments
# ------------------------------------------------------------

export AbstractEnv, AbstractEnvWrapper, AbstractParallelEnv, AbstractParallelEnvWrapper
export act!, observe, observe!, reset!, step!, terminated, truncated
export action_space, get_info, number_of_envs, observation_space
export is_wrapper, unwrap, unwrap_all
export allocate_observations, observation_slot

"""
    AbstractEnv

Abstract base type for single reinforcement learning environments.

Subtypes must implement:
- `reset!(env; seed = nothing)`: reset to an initial state, reseeding the env's RNG when
  `seed` is an integer. Returns `nothing`.
- `act!(env, action)`: take an action and return the reward.
- `observe(env)`: return the current observation.
- `terminated(env)`, `truncated(env)`: whether the episode has ended.
- `action_space(env)`, `observation_space(env)`.

Optional:
- `observe!(dest, env)`: write the current observation into `dest`. The default copies
  `observe(env)`; envs with large observations can override it to avoid an allocation.
- `get_info(env)`: extra per-step information. Defaults to `nothing`.

# Ownership
`observe` returns an array the caller may keep: an env must not mutate an array after
returning it. Callers never mutate what an env returns, and envs never mutate the
actions they are given.
"""
abstract type AbstractEnv end

"""
    AbstractParallelEnv

Abstract base type for vectorized environments that step `number_of_envs(penv)` envs
at once. It is a separate hierarchy from [`AbstractEnv`](@ref).

Subtypes must implement:
- `reset!(penv; seed = nothing) -> obs`: reset all envs. With an integer `seed`, env
  `i` is reset with seed `seed + i - 1`.
- `step!(penv, actions) -> (obs, rewards, terminated, truncated, final_obs, infos)`.
- `observe(penv) -> obs`: the current batched observation.
- `number_of_envs(penv)`, `action_space(penv)`, `observation_space(penv)`.

# `step!` results
- `actions` is an `AbstractVector` with one env-space action per env.
- `obs` is a batched array of size `(size(observation_space(penv))..., n_envs)`. Envs that
  finished this step have already been reset, so their columns hold the first
  observation of the next episode.
- `rewards`, `terminated` and `truncated` are vectors of length `n_envs`.
- `final_obs` has the same shape as `obs`. For envs that finished this step, its column
  holds the last observation of the finished episode. Other columns are unspecified.
- `infos` is `nothing`, or a vector with one entry per env.

The ownership rule of [`AbstractEnv`](@ref) applies: callers may keep the returned arrays
and never mutate them.
"""
abstract type AbstractParallelEnv end

"""
    reset!(env::AbstractEnv; seed = nothing) -> nothing
    reset!(penv::AbstractParallelEnv; seed = nothing) -> obs

Reset an environment. An integer `seed` reseeds the env's random state first, so the
episodes that follow are reproducible. A parallel env returns the batched observation.
"""
function reset! end

"""
    act!(env::AbstractEnv, action) -> reward

Take an action in a single environment and return the reward.
"""
function act! end

"""
    observe(env::AbstractEnv) -> observation
    observe(penv::AbstractParallelEnv) -> batched observation

Return the current observation. The caller may keep the result and must not mutate it.
"""
function observe end

"""
    observe!(dest, env::AbstractEnv) -> dest

Write the current observation of `env` into `dest`. The default copies `observe(env)`.
"""
function observe!(dest, env::AbstractEnv)
    dest .= observe(env)
    return dest
end

"""
    terminated(env::AbstractEnv) -> Bool

Whether the episode has reached a terminal state.
"""
function terminated end

"""
    truncated(env::AbstractEnv) -> Bool

Whether the episode was cut short, for example by a time limit.
"""
function truncated end

"""
    action_space(env) -> AbstractSpace

The action space of a single or parallel environment.
"""
function action_space end

"""
    observation_space(env) -> AbstractSpace

The observation space of a single or parallel environment. For a parallel env, this is
the space of one env's observation.
"""
function observation_space end

"""
    get_info(env::AbstractEnv)

Extra information about the last step. Defaults to `nothing`.
"""
get_info(::AbstractEnv) = nothing

"""
    number_of_envs(penv::AbstractParallelEnv) -> Int

The number of environments stepped by a parallel environment.
"""
function number_of_envs end

"""
    step!(penv::AbstractParallelEnv, actions) -> (obs, rewards, terminated, truncated, final_obs, infos)

Step all environments of `penv` once, resetting the ones that finish. See
[`AbstractParallelEnv`](@ref) for the meaning of each result.
"""
step!

# ------------------------------------------------------------
# Batched observations
# ------------------------------------------------------------

"""
    allocate_observations(space, n) -> Array

An uninitialized array for `n` observations from `space`, of size `(size(space)..., n)`.
"""
function allocate_observations(space, n::Integer)
    return Array{eltype(space)}(undef, size(space)..., n)
end

"""
    observation_slot(obs, i)

A view of the `i`-th observation in the batched array `obs`.
"""
observation_slot(obs::AbstractArray, i::Integer) = selectdim(obs, ndims(obs), i)

# ------------------------------------------------------------
# Environment wrappers
# ------------------------------------------------------------

"""
    AbstractEnvWrapper{E}

Wraps a single [`AbstractEnv`](@ref) while remaining an `AbstractEnv`.
"""
abstract type AbstractEnvWrapper{E <: AbstractEnv} <: AbstractEnv end

"""
    AbstractParallelEnvWrapper{E}

Wraps an [`AbstractParallelEnv`](@ref) (e.g. normalization or monitoring) while remaining
an `AbstractParallelEnv`.
"""
abstract type AbstractParallelEnvWrapper{E <: AbstractParallelEnv} <: AbstractParallelEnv end

"""
    is_wrapper(env) -> Bool

Whether `env` wraps another environment.
"""
is_wrapper(env::AbstractEnv) = env isa AbstractEnvWrapper
is_wrapper(env::AbstractParallelEnv) = env isa AbstractParallelEnvWrapper

"""
    unwrap(env) -> env

Remove one layer of wrapping.
"""
function unwrap end

"""
    unwrap_all(env) -> env

Remove all layers of wrapping.
"""
function unwrap_all(env::Union{AbstractEnv, AbstractParallelEnv})
    while is_wrapper(env)
        env = unwrap(env)
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
    Box{T <: Number, N} <: AbstractSpace

A continuous space with lower and upper bounds per element.

# Fields
- `low::Array{T, N}`: lower bounds
- `high::Array{T, N}`: upper bounds
- `shape::NTuple{N, Int}`: shape of one sample

# Example
```julia
# 2D box with different bounds per dimension
space = Box(Float32[-1, -2], Float32[1, 3])

# Uniform bounds
space = Box(-1.0f0, 1.0f0, (4,))
```
"""
struct Box{T <: Number, N} <: AbstractSpace
    low::Array{T, N}
    high::Array{T, N}
    shape::NTuple{N, Int}
    function Box{T, N}(low::Array{T, N}, high::Array{T, N}) where {T <: Number, N}
        @assert size(low) == size(high) "Low and high arrays must have the same shape"
        @assert all(low .<= high) "All low values must be <= corresponding high values"
        return new{T, N}(low, high, size(low))
    end
end

Box{T}(low::Array{T, N}, high::Array{T, N}) where {T <: Number, N} = Box{T, N}(low, high)
Box(low::Array{T, N}, high::Array{T, N}) where {T <: Number, N} = Box{T, N}(low, high)

function Box(low::T, high::T, shape::NTuple{N, Int}) where {T <: Number, N}
    return Box{T, N}(fill(low, shape), fill(high, shape))
end

Base.ndims(::Box{T, N}) where {T, N} = N

Base.eltype(::Box{T}) where {T} = T

function Base.isequal(box1::Box{T1}, box2::Box{T2}) where {T1, T2}
    return T1 == T2 && box1.low == box2.low && box1.high == box2.high
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
    unit_random = rand(rng, T, space.shape...)
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
    sample in space::Box{T, N}

Whether `sample` is an array of element type `T` and shape `space.shape` that lies within
the bounds. Anything else, including a batch of samples, is not in the space.

# Examples
```julia
space = Box(Float32[-1.0, -2.0], Float32[1.0, 3.0])
Float32[0.5, 1.5] in space  # true
Float32[1.5, 0.0] in space  # false: first element out of bounds
[0.5, 1.5] in space         # false: Float64 sample in a Float32 box
```
"""
function Base.in(sample::AbstractArray{T, N}, space::Box{T, N}) where {T <: Number, N}
    size(sample) == space.shape || return false
    return all(space.low .<= sample .<= space.high)
end

# Dense CPU arrays: a branch-free loop, which does not allocate and vectorizes.
function Base.in(sample::Array{T, N}, space::Box{T, N}) where {T <: Number, N}
    size(sample) == space.shape || return false
    low, high = space.low, space.high
    ok = true
    for i in eachindex(sample, low, high)
        ok &= (low[i] <= sample[i]) & (sample[i] <= high[i])
    end
    return ok
end

Base.in(sample, ::Box) = false

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
    RandomPolicy(env)

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

function RandomPolicy(env::Union{AbstractEnv, AbstractParallelEnv})
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
