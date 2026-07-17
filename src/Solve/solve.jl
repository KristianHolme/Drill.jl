function _done(cache::RLCache)
    return cache.retcode == SciMLBase.ReturnCode.Success ||
        cache.retcode == SciMLBase.ReturnCode.Terminated ||
        cache.steps_taken >= cache.max_steps
end

function CommonSolve.solve!(cache::RLCache)
    if !_callbacks_continue(cache.callbacks, on_training_start, cache)
        cache.retcode = SciMLBase.ReturnCode.Terminated
    end
    while !_done(cache)
        CommonSolve.step!(cache)
    end
    if cache.retcode == SciMLBase.ReturnCode.Default && cache.steps_taken >= cache.max_steps
        cache.retcode = SciMLBase.ReturnCode.Success
    end
    if !_callbacks_continue(cache.callbacks, on_training_end, cache)
        cache.retcode = SciMLBase.ReturnCode.Terminated
    end
    if cache.verbose >= 2
        print_timer(cache.timer)
    end
    return RLSolution(cache)
end

function CommonSolve.solve(prob::RLProblem, alg::AbstractAlgorithm; kwargs...)
    cache = CommonSolve.init(prob, alg; kwargs...)
    return CommonSolve.solve!(cache)
end
