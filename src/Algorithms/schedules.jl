"""
    AbstractSchedule

A hyperparameter value that changes during training. A schedule `s` is called as `s(p)`
with the training progress `p`, the fraction of `max_steps` taken so far: 0 at the start,
1 at the end. Any function `p -> value` can be used in place of a schedule.

Load UnicodePlots to show schedules as a plot in the REPL.

See [`LinearSchedule`](@ref), [`CosineSchedule`](@ref), [`ExponentialSchedule`](@ref).
"""
abstract type AbstractSchedule end

"""
    LinearSchedule(initial, final)

Goes linearly from `initial` at the start of training to `final` at the end.
"""
struct LinearSchedule{T <: AbstractFloat} <: AbstractSchedule
    initial::T
    final::T
end
LinearSchedule(initial::Real, final::Real) = LinearSchedule(promote(float(initial), float(final))...)

(s::LinearSchedule)(p::Real) = s.initial + (s.final - s.initial) * p

"""
    CosineSchedule(initial, final)

Cosine annealing from `initial` at the start of training to `final` at the end:
`final + (initial - final) * (1 + cos(π p)) / 2`.
"""
struct CosineSchedule{T <: AbstractFloat} <: AbstractSchedule
    initial::T
    final::T
end
CosineSchedule(initial::Real, final::Real) = CosineSchedule(promote(float(initial), float(final))...)

(s::CosineSchedule)(p::Real) = s.final + (s.initial - s.final) * (1 + cospi(p)) / 2

"""
    ExponentialSchedule(initial, final)

Decays geometrically from `initial` at the start of training to `final` at the end:
`initial * (final / initial)^p`. Both values must be positive.
"""
struct ExponentialSchedule{T <: AbstractFloat} <: AbstractSchedule
    initial::T
    final::T
    function ExponentialSchedule{T}(initial::T, final::T) where {T <: AbstractFloat}
        initial > 0 && final > 0 ||
            throw(ArgumentError("ExponentialSchedule needs positive values, got $initial and $final"))
        return new{T}(initial, final)
    end
end
function ExponentialSchedule(initial::Real, final::Real)
    initial, final = promote(float(initial), float(final))
    return ExponentialSchedule{typeof(initial)}(initial, final)
end

(s::ExponentialSchedule)(p::Real) = s.initial * (s.final / s.initial)^p

"""
    schedule_value(x, p)

The value of `x` at training progress `p`: `x` itself for a number, `x(p)` for a schedule
or function.
"""
schedule_value(x::Real, p::Real) = x
schedule_value(f, p::Real) = f(p)

"""
    is_constant(x)

Whether `x` is a plain number rather than a schedule.
"""
is_constant(::Real) = true
is_constant(_) = false

_schedule_name(s::AbstractSchedule) = nameof(typeof(s))

function Base.show(io::IO, s::Union{LinearSchedule, CosineSchedule, ExponentialSchedule})
    print(io, _schedule_name(s), "(", s.initial, ", ", s.final, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", s::AbstractSchedule)
    ext = Base.get_extension(parentmodule(@__MODULE__), :Drill_UnicodePlotsExt)
    if ext === nothing
        show(io, s)
    else
        ext.show_schedule_plot(io, s)
    end
    return nothing
end
