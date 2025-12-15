# Getting Started

## Quick Example: PPO on CartPole

```julia
using DRiL
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
using DRiL
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

