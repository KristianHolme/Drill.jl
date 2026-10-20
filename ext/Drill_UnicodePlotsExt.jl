module Drill_UnicodePlotsExt

using Drill: Drill, AbstractSchedule
using UnicodePlots: UnicodePlots
import UnicodePlots: lineplot

"""
    lineplot(schedule::AbstractSchedule; n = 101, kwargs...)

Plot `schedule` against training progress from 0 to 1, evaluated at `n` points. Other
keyword arguments go to `UnicodePlots.lineplot`.
"""
function lineplot(s::AbstractSchedule; n::Integer = 101, kwargs...)
    progress = range(0.0, 1.0; length = n)
    return lineplot(
        progress, [s(p) for p in progress];
        title = sprint(show, s), xlabel = "progress", xlim = (0, 1),
        width = 40, height = 8, kwargs...,
    )
end

function show_schedule_plot(io::IO, s::AbstractSchedule)
    show(io, MIME("text/plain"), lineplot(s))
    return nothing
end

end
