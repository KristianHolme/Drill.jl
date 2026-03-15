```@raw html
---
layout: home

hero:
  name: "Drill.jl"
  text: "Deep Reinforcement Learning in Julia"
  tagline: Fast, flexible, and easy-to-use deep RL algorithms
  image:
    src: /logo.png
    alt: Drill.jl
  actions:
    - theme: brand
      text: Getting Started
      link: /getting_started
    - theme: alt
      text: API Reference
      link: /api
    - theme: alt
      text: View on GitHub
      link: https://github.com/KristianHolme/Drill.jl

features:
  - icon: 🚀
    title: Fast & Extensible
    details: Built on Lux.jl for efficient neural networks with automatic differentiation. Optional Reactant support for GPU/accelerator execution. Pure Julia for easy customization.
  - icon: 🎮
    title: Flexible Environments
    details: Comprehensive interface supporting discrete and continuous action spaces with parallel execution.
  - icon: 📊
    title: Rich Logging
    details: TensorBoard and Weights & Biases integration for real-time training monitoring and analysis.
---
```

## Implemented Algorithms

| Algorithm                                      | Type        |
|------------------------------------------------|-------------|
| **PPO** (Proximal Policy Optimization)         | On-policy   |
| **SAC** (Soft Actor-Critic)                    | Off-policy  |

## How to Install Drill.jl?

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/KristianHolme/Drill.jl")
```

## Quick Example

```julia
using Drill
using Zygote  # Required for automatic differentiation
using ClassicControlEnvironments

# Parallel environments
env = BroadcastedParallelEnv([CartPoleEnv() for _ in 1:4])

# Actor-Critic network
model = ActorCriticLayer(observation_space(env), action_space(env))

# Train with PPO
agent = Agent(model, PPO(); verbose=2)
train!(agent, env, PPO(), 100_000)

# Extract deployment policy
policy = extract_policy(agent)

# Use policy for inference
obs = observe(env)
actions = policy(obs; deterministic=true)
```
