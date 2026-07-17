function _record_stat!(cache::RLCache, key::Symbol, value)
    values = get!(cache.stats, key) do
        typeof(value)[]
    end
    push!(values, value)
    return values
end

function _mark_complete!(cache::RLCache)
    if cache.steps_taken >= cache.max_steps && cache.retcode != ReturnCode.Terminated
        cache.retcode = ReturnCode.Success
    end
    return cache
end

function step!(cache::RLCache{<:Any, <:PPO})
    return train_step!(cache, cache.alg)
end

function step!(cache::RLCache{<:Any, <:SAC})
    return train_step!(cache, cache.alg)
end
