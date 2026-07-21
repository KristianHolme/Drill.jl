"""
    Verbosity

Console verbosity controls for training.

# Fields
- `meter::Int`: ProgressMeter level — `0` off, `1` step progress, `2` step progress with live stats.
- `table::Bool`: Print a PrettyTables summary of the latest stats (requires PrettyTables loaded).
- `timer::Int`: TimerOutputs level — `0` disabled (`NoTimerOutput`, zero-overhead `@timeit`),
  `1` enabled without printing, `2` enabled and print at the end of `solve!`.

Construct from a `NamedTuple` (merged with defaults), an `Integer` meter shorthand
(`table` false / `timer` 0), or a `Verbosity` value via [`normalize_verbosity`](@ref).
"""
Base.@kwdef struct Verbosity
    meter::Int = 2
    table::Bool = false
    timer::Int = 0
end

const DEFAULT_VERBOSITY = (; meter = 2, table = false, timer = 0)

"""
    normalize_verbosity(v) -> Verbosity

Normalize a verbosity specification to a [`Verbosity`](@ref).

Accepts:
- a [`Verbosity`](@ref) value
- a `NamedTuple` merged with defaults `(; meter = 2, table = false, timer = 0)`
- an `Integer` meter shorthand with `table = false` and `timer = 0`

Meter and timer values above `2` warn and clamp to `2`. Negative values throw.
"""
function normalize_verbosity end

function clamp_meter(meter::Integer)
    m = Int(meter)
    m < 0 && throw(ArgumentError("verbosity.meter must be >= 0, got $m"))
    if m > 2
        @warn "verbosity.meter max level is 2; clamping $m to 2"
        return 2
    end
    return m
end

function clamp_timer(timer::Integer)
    t = Int(timer)
    t < 0 && throw(ArgumentError("verbosity.timer must be >= 0, got $t"))
    if t > 2
        @warn "verbosity.timer max level is 2; clamping $t to 2"
        return 2
    end
    return t
end

function normalize_verbosity(v::Verbosity)
    return Verbosity(clamp_meter(v.meter), v.table, clamp_timer(v.timer))
end

function normalize_verbosity(v::NamedTuple)
    return normalize_verbosity(Verbosity(; merge(DEFAULT_VERBOSITY, v)...))
end

function normalize_verbosity(meter::Integer)
    return normalize_verbosity(Verbosity(; meter = Int(meter), table = false, timer = 0))
end

"""
    print_training_table(cache) -> Nothing

Print the latest training metrics as a table. Implemented by `Drill_PrettyTablesExt`
when PrettyTables is loaded. The default method errors if `verbosity.table` is used
without PrettyTables.
"""
function print_training_table(cache)
    throw(
        ArgumentError(
            "verbosity.table=true requires PrettyTables.jl. Add and load PrettyTables to enable Drill_PrettyTablesExt."
        )
    )
end
