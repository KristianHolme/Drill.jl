# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------
"""
    AbstractCallback

Hook type for training: implement any of [`on_training_start`](@ref), [`on_rollout_start`](@ref), [`on_rollout_end`](@ref), [`on_step`](@ref), [`on_training_end`](@ref). Each receives `(callback, locals::Dict)` and must return `true` to continue or `false` to stop training.

`locals` is built with `Base.@locals` inside [`train!`](@ref) and contains loop variables (agent, env, step counters, etc.).
"""
abstract type AbstractCallback end
