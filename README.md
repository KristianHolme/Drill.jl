# DRiL.jl

[![Build Status](https://github.com/KristianHolme/DRiL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KristianHolme/DRiL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://KristianHolme.github.io/DRiL.jl/dev)

**Deep ReInforcement Learning** - A (aspirationally) high-performance Julia package for deep reinforcement learning algorithms.

## Overview

DRiL.jl is a prototype DRL package, aiming to be fast, flexible, and easy to use.

## Main Features

  
- **Modern Architecture**: Built on [Lux.jl](https://github.com/LuxDL/Lux.jl) for neural networks with automatic differentiation support
- **Flexible Environments**: Comprehensive environment interface supporting both discrete and continuous action spaces
- **Rich Logging**: TensorBoard and WandB integration for training monitoring, and timer output ([TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl)) for performance analysis
- **Parallelization**: Built-in support for parallel environment execution

## Implemented Algorithms

- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)

## Core Components
The DRiL.jl package is built around the following core components: **Environments**, **Layers**, **Agents**, and **Algorithms**.
The environment is the system we are interested in controlling, the layer is the training-time actor–critic network used for control, the agent manages training, and the algorithm specifies the training procedure and loss.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/KristianHolme/DRiL.jl")
```

## Quick Start Example

Here's a complete example training a PPO agent on the CartPole environment:

```julia
using DRiL
using Pkg
Pkg.add(url="https://github.com/KristianHolme/ClassicControlEnvironments.jl")
using ClassicControlEnvironments
using Random

## Environment
parallel_env = MultiThreadedParallelEnv([CartPoleEnv() for _ in 1:4])

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

## Agent
agent = Agent(model, ppo; verbose=2)

## Train
max_steps = 100_000
learn_stats, to = train!(agent, parallel_env, ppo, max_steps)

## Evaluate the trained agent
eval_env = CartPoleEnv(max_steps=500)
eval_stats = evaluate_agent(agent, eval_env, n_episodes=10, deterministic=true)

println("Average episodic return: $(mean(eval_stats.episodic_returns))")
println("Average episode length: $(mean(eval_stats.episodic_lengths))")

# Print timer output
print_timer(to)
```

## Advanced Usage

### Custom Environments

Implement the DRiL environment interface:

```julia
struct MyEnv <: AbstractEnv
    # Your environment state
end

# Required methods
DRiL.reset!(env::MyEnv) = # Reset environment
DRiL.act!(env::MyEnv, action) = # Take action, return reward  
DRiL.observe(env::MyEnv) = # Return current observation
DRiL.terminated(env::MyEnv) = # Check if episode is done
DRiL.truncated(env::MyEnv) = # Check if episode is truncated
DRiL.action_space(env::MyEnv) = # Return action space
DRiL.observation_space(env::MyEnv) = # Return observation space
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

### Custom Layer Architectures

```julia
model = ActorCriticLayer(
    obs_space,
    act_space,
    hidden_dims=[128, 128, 64],  # Larger network
    activation=relu,              # Different activation
)
```

### Deployment (lightweight policy)

```julia
# Extract a deployment-time policy (actor-only)
dp = extract_policy(agent)

# Predict env-ready actions
env_actions = predict(dp, batch_of_obs; deterministic=true)
```

## Benchmarking (AirspeedVelocity.jl)

Benchmarks live in `benchmark/benchmarks.jl` and are used by the CI workflow
`.github/workflows/benchmarks.yml`.

Run locally:

```bash
julia -e 'using Pkg; Pkg.add("AirspeedVelocity")'
mkdir -p benchmark_results
benchpkg \
  --path . \
  --rev dirty,main \
  --script benchmark/benchmarks.jl \
  --output-dir benchmark_results \
  --add https://github.com/KristianHolme/ClassicControlEnvironments.jl,Zygote
```