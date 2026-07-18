# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------
"""
    AbstractCallback

Hook type for training: implement any of [`on_training_start`](@ref), [`on_rollout_start`](@ref), [`on_rollout_end`](@ref), [`on_step`](@ref), [`on_training_end`](@ref). Each receives `(callback, cache)` and must return `true` to continue or `false` to stop training.

The `cache` is the active `RLCache` and contains the problem, algorithm, counters, buffer, logger, and train state.
"""
abstract type AbstractCallback end

"""
    on_training_start(callback, cache) -> Bool

Called once at the start of training. Default: `true` (continue).
"""
function on_training_start(callback::AbstractCallback, cache)
    return true
end

"""
    on_rollout_start(callback, cache) -> Bool

Called at the beginning of each rollout collection phase. Default: `true`.
"""
function on_rollout_start(callback::AbstractCallback, cache)
    return true
end

"""
    on_step(callback, cache) -> Bool

Optional per-step hook when algorithms emit it (default implementation: `true`). Override in subtypes as needed.
"""
function on_step(callback::AbstractCallback, cache)
    return true
end

"""
    on_rollout_end(callback, cache) -> Bool

Called after rollout data is collected, before gradient updates. Default: `true`.
"""
function on_rollout_end(callback::AbstractCallback, cache)
    return true
end

"""
    on_training_end(callback, cache) -> Bool

Called when training finishes normally. Default: `true`.
"""
function on_training_end(callback::AbstractCallback, cache)
    return true
end
