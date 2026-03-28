"""
    on_training_start(callback, locals) -> Bool

Called once at the start of [`train!`](@ref). Default: `true` (continue).
"""
function on_training_start(callback::AbstractCallback, locals::Dict)
    return true
end

"""
    on_rollout_start(callback, locals) -> Bool

Called at the beginning of each rollout collection phase. Default: `true`.
"""
function on_rollout_start(callback::AbstractCallback, locals::Dict)
    return true
end

"""
    on_step(callback, locals) -> Bool

Optional per-step hook when algorithms emit it (default implementation: `true`). Override in subtypes as needed.
"""
function on_step(callback::AbstractCallback, locals::Dict)
    return true
end

"""
    on_rollout_end(callback, locals) -> Bool

Called after rollout data is collected, before gradient updates. Default: `true`.
"""
function on_rollout_end(callback::AbstractCallback, locals::Dict)
    return true
end

"""
    on_training_end(callback, locals) -> Bool

Called when training finishes normally. Default: `true`.
"""
function on_training_end(callback::AbstractCallback, locals::Dict)
    return true
end
