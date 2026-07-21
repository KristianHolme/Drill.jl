# Drill.jl

[![Build Status](https://github.com/KristianHolme/Drill.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KristianHolme/Drill.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://KristianHolme.github.io/Drill.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://KristianHolme.github.io/Drill.jl/dev)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![DOI](https://zenodo.org/badge/978281603.svg)](https://doi.org/10.5281/zenodo.19501897)

## Overview

Drill.jl is an experimental deep reinforcement learning package, aiming to be fast, flexible, and easy to use.

## Main Features

- **Modern Architecture**: Built on [Lux.jl](https://github.com/LuxDL/Lux.jl) for neural networks with automatic differentiation support
- **Flexible Environments**: Comprehensive environment interface supporting both discrete and continuous action spaces
- **Rich Logging**: TensorBoard and WandB integration for training monitoring, and timer output ([TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl)) for performance analysis
- **Parallelization**: Built-in support for parallel environment execution

## Implemented Algorithms

- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)

## Core Components

The Drill.jl package is built around the following core components: **Environments**, **Layers**, **Algorithms**, and **Agents**.
The environment is the system we are interested in controlling, implementing the [DrillInterface.jl](https://github.com/KristianHolme/Drill.jl/tree/main/DrillInterface) interface.
The layer is a Lux Layer and contains the neural network(s) defining and required for training, the policy.
The algorithm specifies the training procedure and loss function(s), and the agent manages training of the layer parameters according to the algorithm.

## Installation

```julia
using Pkg
Pkg.add("Drill")
```

## Testing

Run tests from the package project:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Wandb tests use a shared CondaPkg environment by default (`@drill-wandb-tests`), so local
runs reuse the same Conda/Python environment across sessions. For a project-local cache
instead, set:

```bash
JULIA_CONDAPKG_ENV="$PWD/.condapkg/wandb" julia --project=. -e 'using Pkg; Pkg.test()'
```

Avoid using a path containing `.CondaPkg` for `JULIA_CONDAPKG_ENV`, since CondaPkg reserves
that name for project-local environments.

## Quick Start Example

Here's a complete example training a PPO agent on the CartPole environment:

```julia
using Drill
using Pkg
Pkg.add("ClassicControlEnvironments")
using ClassicControlEnvironments
using Random

## Environment
parallel_env = BroadcastedParallelEnv([CartPoleEnv() for _ in 1:4])

## Actor-Critic Layer
model = ActorCriticLayer(
    observation_space(parallel_env),
    action_space(parallel_env)
)

## Algorithm
ppo = PPO(
    gamma=0.99f0,
    gae_lambda=0.95f0,
    clip_range=0.2f0,
    ent_coef=0.01f0,
    vf_coef=0.5f0,
    normalize_advantage=true
)

## Train
max_steps = 100_000
prob = RLProblem(parallel_env, model)
cache = init(
    prob, ppo;
    max_steps,
    # meter: 0=off, 1=progress bar, 2=progress bar + live stats
    # table: PrettyTables dump each update (requires PrettyTables loaded)
    # timer: 0=off (zero overhead), 1=record, 2=record + print at end
    verbosity = (; meter = 2, table = false, timer = 0),
)
solve!(cache)

## Evaluate the trained agent
eval_env = CartPoleEnv(max_steps=500)
eval_stats = evaluate(extract_policy(cache), eval_env; n_eval_episodes=10, deterministic=true)

println("Average episodic return: $(eval_stats.mean_reward)")
println("Average episode length: $(eval_stats.mean_length)")
```

## Advanced Usage

### Custom Environments

When implementing an environment, depend on **DrillInterface** (which also provides `check_env`); add **Drill** when you need training or wrappers. Implement the Drill environment interface:

```julia
struct MyEnv <: AbstractEnv
    # Your environment state
end

# Required methods
DrillInterface.reset!(env::MyEnv) = # Reset environment
DrillInterface.act!(env::MyEnv, action) = # Take action, return reward
DrillInterface.observe(env::MyEnv) = # Return current observation
DrillInterface.terminated(env::MyEnv) = # Check if episode is done
DrillInterface.truncated(env::MyEnv) = # Check if episode is truncated
DrillInterface.action_space(env::MyEnv) = # Return action space
DrillInterface.observation_space(env::MyEnv) = # Return observation space
```

### Environment Wrappers

```julia
# Normalize observations and rewards
env = NormalizeWrapperEnv(env, normalize_obs=true, normalize_reward=true)

# Monitor episode statistics
env = MonitorWrapperEnv(env)

# Scale observations and actions
env = ScalingWrapperEnv(env)
```

### AD backends and Device support (CPU / GPU)

Different backends for automatic differentiation are supported through the `ad_type` keyword argument to the `train!` function. Currently, [Zygote.jl](https://github.com/FluxML/Zygote.jl) is the default (using the `AutoZygote()` backend). [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) is also supported by using the `AutoEnzyme()` backend. For the SAC algorithm, runtime activity must be turned on (`AutoEnzyme(; mode = set_runtime_activity(Reverse))`). The corresponding package (Zygote/Enzyme) must be loaded before calling `train!`.

[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) support is experimental: put parameters on `reactant_device()` and train with `AutoEnzyme()`. Lux's `TrainState` owns training compilation; Drill only `@compile`s rollout/deployment inference kernels. Multi-objective algorithms (SAC) use one Lux `TrainState` per objective (actor / critic / entropy).

### Custom Layer Architectures

```julia
model = ActorCriticLayer(
    obs_space,
    act_space,
    hidden_dims=[128, 128, 64],  # Larger network
    activation=relu,              # Different activation
)
```

## Benchmarking with [AirSpeedVelocity.jl](https://github.com/MilesCranmer/AirSpeedVelocity.jl)

Benchmarks live in `benchmark/benchmarks.jl`.

### CI

- **Basic suite** (automatic on every PR): `rollouts`, `training`, and `wrappers` via [`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml). Heavy packages (Zygote, Enzyme, Reactant) are not installed for this path.
- **Full suite** (on demand): comment `/bench-full` on a PR (OWNER/MEMBER/COLLABORATOR), or run the **Full benchmarks** workflow from the Actions tab. This includes `devices` and `ad_backends` and installs Zygote, Enzyme, and Reactant. See [`.github/workflows/benchmarks-full.yml`](.github/workflows/benchmarks-full.yml).

### Local

First install and build [AirSpeedVelocity.jl](https://github.com/MilesCranmer/AirSpeedVelocity.jl) as described in its readme, then:

Basic suite (matches PR CI):

```bash
mkdir -p benchmark_results
benchpkg \
  --path . \
  --rev dirty,main \
  --script benchmark/benchmarks.jl \
  --output-dir benchmark_results \
  --filter rollouts,training,wrappers \
  --add ClassicControlEnvironments
```

Full suite:

```bash
mkdir -p benchmark_results
benchpkg \
  --path . \
  --rev dirty,main \
  --script benchmark/benchmarks.jl \
  --output-dir benchmark_results \
  --add ClassicControlEnvironments,Zygote,Enzyme,Reactant
```
