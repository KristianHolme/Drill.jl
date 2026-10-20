# Algorithms


## PPO (Proximal Policy Optimization)

On-policy algorithm using clipped surrogate objective.

```julia
using Drill
using Zygote
using ClassicControlEnvironments
env = BroadcastedParallelEnv([CartPoleEnv() for _ in 1:4])
obs_space = state_space(env)
act_space = action_space(env)
max_steps = 100_000

ppo = PPO(;
    gamma = 0.99f0,
    gae_lambda = 0.95f0,
    clip_range = 0.2f0,
    ent_coef = 0.0f0,
    vf_coef = 0.5f0,
    max_grad_norm = 0.5f0,
    n_steps = 2048,
    batch_size = 64,
    epochs = 10,
    learning_rate = 3f-4,
)

model = ActorCriticLayer(obs_space, act_space)
agent = Agent(model, ppo)
train!(agent, env, ppo, max_steps)
```

## SAC (Soft Actor-Critic)

Off-policy algorithm with entropy regularization and twin Q-networks.

```julia
using Drill
using Zygote
using ClassicControlEnvironments
env = BroadcastedParallelEnv([CartPoleEnv() for _ in 1:4])
obs_space = state_space(env)
act_space = action_space(env)
max_steps = 100_000

sac = SAC(;
    learning_rate = 3f-4,
    buffer_capacity = 1_000_000,
    batch_size = 256,
    tau = 0.005f0,
    gamma = 0.99f0,
    train_freq = 1,
    gradient_steps = 1,
    ent_coef = AutoEntropyCoefficient(),
)

model = SACLayer(obs_space, act_space)
agent = Agent(model, sac)
train!(agent, env, sac, max_steps)
```

## Entropy Coefficient

```julia
# Fixed entropy
sac = SAC(ent_coef = FixedEntropyCoefficient(0.2f0))

# Automatic tuning (default)
sac = SAC(ent_coef = AutoEntropyCoefficient())
```


## Learning rate schedules

`learning_rate` takes a number or a schedule. A schedule is a function of the training
progress `p`, the fraction of `max_steps` taken so far, from 0 at the start to 1 at the
end. The learning rate is updated after each rollout for PPO and after each collection
step for SAC, before the gradient updates.

```julia
ppo = PPO(; learning_rate = LinearSchedule(3f-4, 0f0))       # linear decay to 0
sac = SAC(; learning_rate = CosineSchedule(1f-3, 1f-5))      # cosine annealing
ppo = PPO(; learning_rate = ExponentialSchedule(1f-3, 1f-5)) # geometric decay
ppo = PPO(; learning_rate = p -> p < 0.1f0 ? 3f-5 : 3f-4)    # any function of p
```

The current value is logged as `train/learning_rate` and recorded in
`cache.stats[:learning_rates]`. With UnicodePlots loaded, a schedule shows as a plot in
the REPL, and `lineplot(schedule)` returns the plot:

```julia
julia> using UnicodePlots

julia> CosineSchedule(1f-3, 0f0)
```
