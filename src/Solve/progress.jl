function latest_stat(cache::RLCache, key::Symbol, default = nothing)
    values = get(cache.stats, key, nothing)
    if values === nothing || isempty(values)
        return default
    end
    return values[end]
end

"""
    training_metric_rows(cache) -> Vector{Pair{String,Any}}

Latest recorded training stats as ordered name => value pairs for console tables.
"""
function training_metric_rows(cache::RLCache)
    rows = Pair{String, Any}[]
    push!(rows, "steps_taken" => cache.steps_taken)
    push!(rows, "gradient_updates" => cache.gradient_updates)
    for key in sort!(collect(keys(cache.stats)); by = string)
        value = latest_stat(cache, key)
        value === nothing && continue
        push!(rows, string(key) => value)
    end
    return rows
end

function update_training_progress!(cache::RLCache, step::Integer; showvalues = nothing)
    meter = cache.verbosity.meter
    progress_meter = cache.progress_meter
    if meter > 0 && progress_meter !== nothing
        if meter >= 2 && showvalues !== nothing
            next!(progress_meter; step = step, showvalues = showvalues)
        else
            next!(progress_meter; step = step)
        end
    end
    if cache.verbosity.table
        print_training_table(cache)
    end
    return nothing
end

function finish_training_progress!(cache::RLCache)
    progress_meter = cache.progress_meter
    if progress_meter !== nothing && cache.verbosity.meter > 0
        ProgressMeter.finish!(progress_meter)
    end
    return nothing
end
