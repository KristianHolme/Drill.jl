# API Reference

## Environments

```@docs
AbstractEnv
AbstractParallelEnv
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
predict_actions
predict_values
steps_taken
evaluate_agent
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
```

## Deployment

```@docs
extract_policy
DeploymentPolicy
NormalizedDeploymentPolicy
```

## Logging

```@docs
AbstractTrainingLogger
NoTrainingLogger
log_scalar!
log_metrics!
set_step!
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

