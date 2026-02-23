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
    details: Built on Lux.jl for efficient neural networks with automatic differentiation. Pure Julia for easy customization.
  - icon: 🎮
    title: Flexible Environments
    details: Comprehensive interface supporting discrete and continuous action spaces with parallel execution.
  - icon: 📊
    title: Rich Logging
    details: TensorBoard and Weights & Biases integration for real-time training monitoring and analysis.
  - icon: ⚡
    title: Production-Ready
    details: Extract lightweight deployment policies. Save and load normalization statistics for consistent inference.
---
```

```@raw html
<div class="vp-doc" style="width:80%; margin:auto">

<h2>Implemented Algorithms</h2>
<table>
<tr><th>Algorithm</th><th>Type</th></tr>
<tr><td><strong>PPO</strong> (Proximal Policy Optimization)</td><td>On-policy</td></tr>
<tr><td><strong>SAC</strong> (Soft Actor-Critic)</td><td>Off-policy</td></tr>
</table>

<h2>How to Install Drill.jl?</h2>
</div>
```

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/KristianHolme/Drill.jl")
```

```@raw html
<div class="vp-doc" style="width:80%; margin:auto">
<h2>Quick Example</h2>
</div>
```

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
