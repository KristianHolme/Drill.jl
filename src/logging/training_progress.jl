"""
    TrainingProgressOptions

Configuration for [`ProgressMeter`](https://github.com/timholy/ProgressMeter.jl) during training.

# Fields
- `output::IO = stderr`: Stream for progress output. Redirect to a tty path when running under a cluster scheduler.
- `color::Symbol = :green`: Progress bar color. Use `:normal` for plain text without ANSI escape codes.
- `valuecolor::Symbol = :blue`: Color for metric values when `verbose > 1`.
- `showspeed::Bool = true`: Show average time per iteration.

Use [`plain_progress_options`](@ref) for file-friendly output on HPC clusters.
"""
Base.@kwdef struct TrainingProgressOptions
    output::IO = stderr
    color::Symbol = :green
    valuecolor::Symbol = :blue
    showspeed::Bool = true
end

"""
    plain_progress_options(; output = stderr, showspeed = true) -> TrainingProgressOptions

Return progress options without ANSI colors, suitable for logging to file.
"""
function plain_progress_options(; output::IO = stderr, showspeed::Bool = true)
    return TrainingProgressOptions(;
        output = output,
        color = :normal,
        valuecolor = :normal,
        showspeed = showspeed,
    )
end

"""
    make_training_progress_meter(total_steps, verbose, options = TrainingProgressOptions()) -> Progress

Create a training progress meter from total steps, agent verbosity, and options.
"""
function make_training_progress_meter(
        total_steps::Integer,
        verbose::Integer,
        options::TrainingProgressOptions = TrainingProgressOptions(),
    )
    return Progress(
        total_steps;
        desc = "Training...",
        showspeed = options.showspeed,
        enabled = verbose > 0,
        output = options.output,
        color = options.color,
    )
end

"""
    progress_next!(progress_meter, options::TrainingProgressOptions; kwargs...) -> Nothing

Update a training progress meter, applying configured colors to metric values.
"""
function progress_next!(
        progress_meter::Progress,
        options::TrainingProgressOptions;
        kwargs...
    )
    return ProgressMeter.next!(progress_meter; valuecolor = options.valuecolor, kwargs...)
end
