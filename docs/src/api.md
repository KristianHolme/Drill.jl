# API Reference

## Environments

```@docs
AbstractEnv
AbstractParallelEnv
AbstractParallelEnvWrapper
reset!
act!
observe
terminated
truncated
action_space
observation_space
get_info
number_of_envs
```

## Spaces

```@docs
AbstractSpace
Box
Discrete
```

## Algorithms

```@docs
PPO
SAC
train!
```

## Agents

```@docs
Agent
AgentStats
predict_actions
predict_values
steps_taken
evaluate_agent
AbstractActionAdapter
to_env
from_env
```

## Layers

```@docs
ActorCriticLayer
ContinuousActorCriticLayer
DiscreteActorCriticLayer
SACLayer
```

## Buffers

```@docs
RolloutBuffer
ReplayBuffer
```

## Wrappers

```@docs
MultiThreadedParallelEnv
BroadcastedParallelEnv
NormalizeWrapperEnv
ScalingWrapperEnv
MonitorWrapperEnv
EpisodeStats
RunningMeanStd
set_training
is_training
sync_normalization_stats!
```

## Deployment

```@docs
extract_policy
NeuralPolicy
NormWrapperPolicy
```

## Logging

```@docs
AbstractTrainingLogger
NoTrainingLogger
log_scalar!
log_metrics!
set_step!
increment_step!
log_hparams!
flush!
close!
```

## Callbacks

```@docs
AbstractCallback
on_training_start
on_training_end
on_rollout_start
on_rollout_end
on_step
```

