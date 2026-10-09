# Environments

## Dependencies for environment implementers

Packages or code that only implement environments should depend on **DrillInterface**, not Drill. DrillInterface is lightweight (minimal dependencies) and provides the types and function signatures needed to implement `AbstractEnv`. Add **Drill** when you need training (PPO, SAC), parallel environment wrappers (`MultiThreadedParallelEnv`, `BroadcastedParallelEnv`), or other wrappers (e.g. `NormalizeWrapperEnv`). Environment validation (`check_env`) is provided by DrillInterface; Drill re-exports it for convenience. DrillInterface is available from the same repository as Drill.

## Interface

Implement `AbstractEnv` with these required methods:

```julia
struct MyEnv <: AbstractEnv
    # state fields
end

DrillInterface.reset!(env::MyEnv; seed = nothing) # Reset; reseed the env's RNG if `seed` is an Int
DrillInterface.act!(env::MyEnv, action)          # Take action, return reward
DrillInterface.observe(env::MyEnv)               # Return current observation
DrillInterface.terminated(env::MyEnv)            # Terminal state reached?
DrillInterface.truncated(env::MyEnv)             # Time limit reached?
DrillInterface.action_space(env::MyEnv)          # Return action space
DrillInterface.observation_space(env::MyEnv)     # Return observation space
```

Optional methods:

```julia
DrillInterface.observe!(dest, env::MyEnv)  # Write the observation into `dest` (default: copy `observe(env)`)
DrillInterface.get_info(env::MyEnv)        # Extra per-step information (default: `nothing`)
```

Ownership: `observe` returns an array the caller may keep, so an env must not mutate an
array after returning it. Callers never mutate what an env returns, and envs never mutate
the actions they are given. Override `observe!` when observations are large and you want
to avoid an allocation per step.

## Spaces

```julia
# Continuous actions/observations: Box{T, N}
Box(low, high)              # low/high are arrays of the same shape
Box(-1.0f0, 1.0f0, (4,))    # uniform bounds with a given shape

# Discrete actions
Discrete(n)              # Actions: 1, 2, ..., n
Discrete(n, start=0)     # Actions: 0, 1, ..., n-1
```

## Parallel Environments

```julia
# Multi-threaded (multi-threaded, keep threading overhead in mind!)
env = MultiThreadedParallelEnv([MyEnv() for _ in 1:n_envs])

# Broadcasted (single-threaded, often fastest for cheap environments like those in ClassicControlEnvironments.jl)
env = BroadcastedParallelEnv([MyEnv() for _ in 1:n_envs])
```

Parallel environments (`AbstractParallelEnv`) are a separate hierarchy from `AbstractEnv`.
They step all envs in one call and reset finished envs automatically:

```julia
obs = reset!(penv; seed = 1)  # env i is reset with seed 1 + i - 1
obs, rewards, terminated, truncated, final_obs, infos = step!(penv, actions)
```

- `actions` has one env action per env.
- `obs` is a batched array of size `(size(observation_space(penv))..., n_envs)`. Finished
  envs are already reset.
- `final_obs` has the same shape. For envs that finished this step, its column holds the
  last observation of the finished episode, used to bootstrap truncated episodes.
- `infos` is `nothing`, or a vector with one entry per env.

To implement your own parallel env, for example one that steps simulations on
distributed workers, subtype `AbstractParallelEnv` and implement `reset!`, `step!`,
`observe`, `number_of_envs`, `action_space` and `observation_space`.

## Environment Validation

```julia
check_env(env)   # Validates a single env
check_env(penv)  # Validates a parallel env
```

