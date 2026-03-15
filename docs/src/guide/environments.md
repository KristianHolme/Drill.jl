# Environments

## Dependencies for environment implementers

Packages or code that only implement environments should depend on **DrillInterface**, not Drill. DrillInterface is lightweight (minimal dependencies) and provides the types and function signatures needed to implement `AbstractEnv`. Add **Drill** when you need training (PPO, SAC), parallel environment wrappers (`MultiThreadedParallelEnv`, `BroadcastedParallelEnv`), other wrappers (e.g. `NormalizeWrapperEnv`), or environment validation (`check_env`). DrillInterface is available from the same repository as Drill.

## Interface

Implement `AbstractEnv` with these required methods:

```julia
struct MyEnv <: AbstractEnv
    # state fields
end

DrillInterface.reset!(env::MyEnv)           # Reset to initial state
DrillInterface.act!(env::MyEnv, action)     # Take action, return reward
DrillInterface.observe(env::MyEnv)          # Return current observation
DrillInterface.terminated(env::MyEnv)       # Terminal state reached?
DrillInterface.truncated(env::MyEnv)        # Time limit reached?
DrillInterface.action_space(env::MyEnv)     # Return action space
DrillInterface.observation_space(env::MyEnv) # Return observation space
```

## Spaces

```julia
# Continuous actions/observations
Box{Float32}(low, high)  # low/high are vectors

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

## Environment Validation

```julia
check_env(env)  # Validates interface implementation
```

