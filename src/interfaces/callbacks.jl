# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------
"""
    AbstractCallback

Hook type for training: implement any of `on_training_start`, `on_rollout_start`, `on_rollout_end`, `on_step`, `on_training_end`. Each receives `(callback, locals::Dict)` and must return `true` to continue or `false` to stop training.

`locals` is built with `Base.@locals` inside `train!` and contains loop variables (agent, env, step counters, etc.).
"""
abstract type AbstractCallback end
