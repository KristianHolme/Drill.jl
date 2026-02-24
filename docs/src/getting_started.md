# Getting Started

## Quick Example: PPO on CartPole

```julia
using Drill
using Zygote  # Required for automatic differentiation
using ClassicControlEnvironments

# Create parallel environments
env = BroadcastedParallelEnv([CartPoleEnv() for _ in 1:4])

# Actor-Critic network
model = ActorCriticLayer(observation_space(env), action_space(env))

# Algorithm with default hyperparameters
ppo = PPO()

# Create agent
agent = Agent(model, ppo; verbose=2)

# Train for 100k steps
learn_stats, timer = train!(agent, env, ppo, 100_000)

# Extract deployment policy
policy = extract_policy(agent)
```

## Quick Example: SAC on Continuous Control

```julia
using Drill
using Zygote  # Required for automatic differentiation

# Continuous action environment
env = BroadcastedParallelEnv([YourContinuousEnv() for _ in 1:4])

# SAC-specific policy
model = SACLayer(observation_space(env), action_space(env))

# SAC algorithm
sac = SAC(learning_rate=3f-4, buffer_capacity=1_000_000)

# Create agent
agent = Agent(model, sac; verbose=1)

# Train
agent, buffer, stats, timer = train!(agent, env, sac, 500_000)
```

## Key Hyperparameters

### PPO

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO clipping range |
| `ent_coef` | 0.0 | Entropy coefficient |
| `n_steps` | 2048 | Steps per rollout |
| `batch_size` | 64 | Minibatch size |
| `epochs` | 10 | Epochs per update |

### SAC

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Soft update rate |
| `buffer_capacity` | 1M | Replay buffer size |
| `batch_size` | 256 | Batch size |
| `train_freq` | 1 | Steps between updates |

## Evaluation

```julia
eval_env = CartPoleEnv(max_steps=500)
stats = evaluate_agent(agent, eval_env; n_episodes=10, deterministic=true)
println("Mean return: $(mean(stats.episodic_returns))")
```

## Deployment

```julia
# Extract lightweight policy (actor only)
policy = extract_policy(agent)

# Get actions for observations
actions = predict(policy, observations; deterministic=true)
```

## Device support (CPU / GPU)

Agents and policies are created on CPU by default. Use the same API as for data: pipe to a device to get a copy on that device. Drill moves data automatically during training and inference; you only choose where the agent (or policy) lives.

```julia
using Lux  # or MLDataDevices: cpu_device, gpu_device

# Move agent to GPU for training
agent = agent |> gpu_device()

# Or pass the device at construction (convenience)
agent = Agent(model, ppo; device = gpu_device())

# Deploy on CPU: extract policy then move to CPU
policy = extract_policy(agent) |> cpu_device()
```

**Performance:** For best performance, Lux recommends **Enzyme** for differentiation and **Reactant** for GPU/accelerator execution. With Reactant loaded, use `agent |> Lux.reactant_device()` to run on the Reactant device. See [Lux's documentation](https://lux.csail.mit.edu/stable/) on compiling models and GPU management for details.

