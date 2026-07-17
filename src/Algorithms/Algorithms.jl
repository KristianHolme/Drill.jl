module Algorithms

using Accessors: @set
import Lux
using Lux: relu
import Optimisers
using Random: AbstractRNG, default_rng
using Statistics: mean, std

import DrillInterface: AbstractSpace, Box, Discrete, action_space

import ..Adapters: ClampAdapter, DiscreteAdapter, TanhScaleAdapter
import ..Models: AbstractActorCriticModel, ContinuousActorCriticModel,
    CriticType, QCritic, VCritic, SeparateFeatures, SharedFeatures,
    action_log_prob, evaluate_actions, predict_values
import ..Buffers: OnPolicyBuffer, OffPolicyBuffer
import ..Utils: discrete_to_onehotbatch, merge_params

include("types.jl")
include("entropy.jl")
include("traits.jl")
include("train_state.jl")
include("ppo.jl")
include("sac.jl")

function train_step! end

export AbstractAlgorithm, OffPolicyAlgorithm, OnPolicyAlgorithm
export AbstractEntropyTarget, AutoEntropyTarget, FixedEntropyTarget
export AbstractEntropyCoefficient, AutoEntropyCoefficient, FixedEntropyCoefficient
export PPO, SAC, SACModel
export train_step!
export AbstractAdvantageStrategy, NormalizeAdvantages, RawAdvantages
export AbstractClipVFStrategy, ClipVF, NoClipVF
export AbstractKLTargetStrategy, KLTarget, NoKLTarget
export normalize_advantage, clip_range_vf, target_kl
export action_adapter, has_twin_critics, has_target_networks, has_entropy_tuning,
    uses_replay, critic_type
export AbstractAlgorithmTrainState, PPOTrainState, SACTrainState
export EntropyCoefficientLayer
export parameters, states, set_states!, lux_train_state, set_lux_train_state!
export entropy_parameters, entropy_coefficient
export select_actor_parameters, select_critic_parameters, select_actor_states,
    select_critic_states, merge_actor_critic_parameters, merge_actor_critic_states,
    project_namedtuple
export make_optimizer, init_entropy_coefficient
export prepare_training_actions, maybe_normalize_batch_data, maybe_clip_range
export process_action
export SACEntropyObjective, SACCriticObjective, SACActorObjective
export sac_actor_loss, sac_critic_loss, compute_target_q_values, get_target_entropy,
    get_gradient_steps
export compatible

end
