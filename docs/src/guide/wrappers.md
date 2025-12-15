# Environment Wrappers

## NormalizeWrapperEnv

Normalizes observations and rewards using running statistics.

```julia
env = NormalizeWrapperEnv(env;
    normalize_obs = true,
    normalize_reward = true,
    clip_obs = 10.0f0,
    clip_reward = 10.0f0,
)

# Save/load normalization stats
save_normalization_stats(env, "norm_stats.jld2")
load_normalization_stats!(env, "norm_stats.jld2")

# Toggle training mode (stops updating stats)
set_training(env, false)
```

## ScalingWrapperEnv

Scales observations and actions to normalized ranges.

```julia
env = ScalingWrapperEnv(env)
```

!!! warning
    ScalingWrapperEnv requires finite bounds on observation and action spaces. It will not work with spaces containing `Inf` or `-Inf` bounds.


## MonitorWrapperEnv

Tracks episode statistics.

```julia
env = MonitorWrapperEnv(env)
# Access stats via get_info(env)
```

## Wrapper Utilities

```julia
# Check if wrapped
is_wrapper(env)

# Unwrap one layer
inner_env = unwrap(env)

# Unwrap all layers
base_env = unwrap_all(env)
```

