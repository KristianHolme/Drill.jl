function _done(cache::RLCache)
    return cache.retcode == ReturnCode.Success ||
        cache.retcode == ReturnCode.Terminated ||
        cache.steps_taken >= cache.max_steps
end

function solve!(cache::RLCache)
    if !_callbacks_continue(cache.callbacks, on_training_start, cache)
        cache.retcode = ReturnCode.Terminated
    end
    while !_done(cache)
        step!(cache)
    end
    if cache.retcode == ReturnCode.Default && cache.steps_taken >= cache.max_steps
        cache.retcode = ReturnCode.Success
    end
    if !_callbacks_continue(cache.callbacks, on_training_end, cache)
        cache.retcode = ReturnCode.Terminated
    end
    finish_training_progress!(cache)
    if cache.verbosity.timer
        print_timer(cache.timer)
    end
    return RLSolution(cache)
end

function solve(prob::RLProblem, alg::AbstractAlgorithm; kwargs...)
    cache = init(prob, alg; kwargs...)
    return solve!(cache)
end
