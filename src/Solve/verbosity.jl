"""
    Verbosity

Console verbosity controls for training.

# Fields
- `meter::Int`: ProgressMeter level — `0` off, `1` step progress, `2` step progress with live stats.
- `table::Bool`: Print a PrettyTables summary of the latest stats (requires PrettyTables loaded).
- `timer::Bool`: Print `TimerOutputs` at the end of `solve!`.

Construct from a `NamedTuple` (merged with defaults), an `Integer` meter shorthand
(`table`/`timer` false), or a `Verbosity` value via [`normalize_verbosity`](@ref).
"""
Base.@kwdef struct Verbosity
    meter::Int = 2
    table::Bool = false
    timer::Bool = true
end

const DEFAULT_VERBOSITY = (; meter = 2, table = false, timer = true)

"""
    normalize_verbosity(v) -> Verbosity

Normalize a verbosity specification to a [`Verbosity`](@ref).

Accepts:
- a [`Verbosity`](@ref) value
- a `NamedTuple` merged with defaults `(; meter = 2, table = false, timer = true)`
- an `Integer` meter shorthand with `table = false` and `timer = false`

Meter values above `2` warn and clamp to `2`. Negative meters throw.
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

function normalize_verbosity(v::Verbosity)
    return Verbosity(clamp_meter(v.meter), v.table, v.timer)
end

function normalize_verbosity(v::NamedTuple)
    return normalize_verbosity(Verbosity(; merge(DEFAULT_VERBOSITY, v)...))
end

function normalize_verbosity(meter::Integer)
    return normalize_verbosity(Verbosity(; meter = Int(meter), table = false, timer = false))
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
