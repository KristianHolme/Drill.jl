# Getting Started

!!! warning "Reactant and GPU"
    [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) integration is experimental. On CPU it is typically **much slower** than training with the usual CPU stack (without Reactant). Prefer the default setup unless you are explicitly testing Reactant or GPU-related workflows.

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

| Parameter    | Default | Description         |
| ------------ | ------- | ------------------- |
| `gamma`      | 0.99    | Discount factor     |
| `gae_lambda` | 0.95    | GAE lambda          |
| `clip_range` | 0.2     | PPO clipping range  |
| `ent_coef`   | 0.0     | Entropy coefficient |
| `n_steps`    | 2048    | Steps per rollout   |
| `batch_size` | 64      | Minibatch size      |
| `epochs`     | 10      | Epochs per update   |

### SAC

| Parameter         | Default | Description           |
| ----------------- | ------- | --------------------- |
| `gamma`           | 0.99    | Discount factor       |
| `tau`             | 0.005   | Soft update rate      |
| `buffer_capacity` | 1M      | Replay buffer size    |
| `batch_size`      | 256     | Batch size            |
| `train_freq`      | 1       | Steps between updates |

## Evaluation

```julia
eval_env = CartPoleEnv(max_steps=500)
stats = evaluate_agent(agent, eval_env; n_episodes=10, deterministic=true)
println("Mean return: $(mean(stats.episodic_returns))")
```

## Extracting a policy from the agent

The agent contains everything needed for producing actions from environment observations, in addition to things needed for training.
To extract only the necessary structures needed for inference, use the `extract_policy` function. The object you get is a `NeuralPolicy`, implementing the minimal policy interface:

```julia
# Extract lightweight policy (actor only)
policy = extract_policy(agent)

observation = observe(env)
# Get action for observation
actions = policy(observation; deterministic=true)
```

If using the @NormalizeWrapperEnv wrapper on the training environment, supply this as the second positional argument to `extract_policy`:

```julia
policy = extract_policy(agent, norm_wrapped_env)
```

This will give you a `NeuralPolicy` wrapped in a `NormWrapperPolicy`, that normalizes the observations you pass to the policy:

```julia
normwrapped_policy = extract_policy(agent, norm_wrapped_training_env)

raw_observation = observe(single_test_time_env)
# Get action for observation
actions = policy(raw_observation; deterministic=true)
```

Here, `single_test_time_env` does not have a normalization wrapper, but the policy applies the normalization needed before obtaining the action from the layer.
