# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------
"""
    AbstractCallback

Hook type for training: implement any of [`on_training_start`](@ref), [`on_rollout_start`](@ref), [`on_rollout_end`](@ref), [`on_step`](@ref), [`on_training_end`](@ref). Each receives `(callback, cache)` and must return `true` to continue or `false` to stop training.

The `cache` is the active `RLCache` and contains the problem, algorithm, counters, buffer, logger, and train state.
"""
abstract type AbstractCallback end
