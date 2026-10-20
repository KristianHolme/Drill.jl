module Drill_UnicodePlotsExt

using Drill: Drill, AbstractSchedule
using UnicodePlots: UnicodePlots
import UnicodePlots: lineplot

"""
    lineplot(schedule::AbstractSchedule; n = 101, kwargs...)

Plot `schedule` against training progress from 0 to 1, evaluated at `n` points. The y axis
spans the schedule's values, or 0 to twice the value for a constant schedule. Other keyword arguments go to `UnicodePlots.lineplot`.
"""
function lineplot(s::AbstractSchedule; n::Integer = 101, kwargs...)
    progress = range(0.0, 1.0; length = n)
    values = [s(p) for p in progress]
    lo, hi = extrema(values)
    # (0, 0) lets UnicodePlots choose
    ylim = lo < hi ? (lo, hi) : hi > 0 ? (zero(hi), 2hi) : (0, 0)
    return lineplot(
        progress, values;
        title = sprint(show, s), xlabel = "progress", xlim = (0, 1), ylim,
        width = 40, height = 8, kwargs...,
    )
end

function show_schedule_plot(io::IO, s::AbstractSchedule)
    show(io, MIME("text/plain"), lineplot(s))
    return nothing
end

end
