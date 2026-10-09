# DrillInterface.jl

Lightweight interface package for reinforcement learning environments. Use DrillInterface when **implementing** environments; depend on [Drill.jl](https://github.com/KristianHolme/Drill.jl) when you need training (PPO, SAC) or parallel env wrappers. DrillInterface provides the interface types and `check_env` for validating implementations.

## Installation

From the Drill.jl repository (same repo as Drill):

```julia
using Pkg
Pkg.add("DrillInterface")
# Or add the parent repo and use the subfolder; see Drill's README.
```

## Implementing an environment

Subtype `AbstractEnv` and implement:

- `reset!(env; seed = nothing)` — reset to initial state, reseeding the env's RNG when `seed` is an integer
- `act!(env, action)` — take action, return reward
- `observe(env)` — current observation
- `terminated(env)` — episode terminated?
- `truncated(env)` — episode truncated (e.g. time limit)?
- `action_space(env)` — action space
- `observation_space(env)` — observation space

Optionally implement `observe!(dest, env)` (default: copy `observe(env)`) and `get_info(env)` (default: `nothing`).

Callers may keep what `observe` returns and never mutate it; envs never mutate the actions they are given.

Spaces: use `Box(low, high)` for continuous and `Discrete(n)` or `Discrete(n, 0)` for discrete actions.

Parallel envs subtype `AbstractParallelEnv` (a separate hierarchy) and implement `reset!(penv; seed) -> obs` and `step!(penv, actions) -> (obs, rewards, terminated, truncated, final_obs, infos)` with batched observations; see the `AbstractParallelEnv` docstring.

```julia
using DrillInterface

struct MyEnv <: AbstractEnv
    # state fields
end

DrillInterface.observation_space(::MyEnv) = Box(Float32[0, 0], Float32[1, 1])
DrillInterface.action_space(::MyEnv) = Discrete(2)
DrillInterface.reset!(env::MyEnv; seed = nothing) = nothing
DrillInterface.act!(env::MyEnv, action) = 0.0f0
DrillInterface.observe(env::MyEnv) = rand(observation_space(env))
DrillInterface.terminated(::MyEnv) = false
DrillInterface.truncated(::MyEnv) = false
```

Validate your implementation with `check_env(env)`. For full documentation and parallel env types, see the [Drill.jl docs](https://KristianHolme.github.io/Drill.jl/dev).
