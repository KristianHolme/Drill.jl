# Algorithm-owned training state bundles (Lux TrainState per objective).

"""
    AbstractAlgorithmTrainState

Algorithm-specific bundle of Lux `TrainState`(s) and related training tensors.
Inference and serialization should go through [`parameters`](@ref) / [`states`](@ref).
"""
abstract type AbstractAlgorithmTrainState end

"""
    PPOTrainState

Single-objective PPO training state wrapping one Lux `TrainState`.
"""
mutable struct PPOTrainState{TS <: Lux.Training.TrainState} <: AbstractAlgorithmTrainState
    ts::TS
end

"""
    EntropyCoefficientLayer

Tiny Lux layer used only as the model handle for the SAC entropy-coefficient `TrainState`.
The objective ignores the layer and reads `ps.log_ent_coef` directly.
"""
struct EntropyCoefficientLayer <: Lux.AbstractLuxLayer end

function Lux.initialparameters(::AbstractRNG, ::EntropyCoefficientLayer)
    return (;)
end

function Lux.initialstates(::AbstractRNG, ::EntropyCoefficientLayer)
    return (;)
end

"""
    SACTrainState

Multi-objective SAC training state: separate Lux `TrainState`s for actor, critic, and
entropy coefficient, plus target-critic parameters/states.
"""
mutable struct SACTrainState{
        ATS <: Lux.Training.TrainState,
        CTS <: Lux.Training.TrainState,
        ETS <: Lux.Training.TrainState,
        TP,
        TST,
    } <: AbstractAlgorithmTrainState
    actor_ts::ATS
    critic_ts::CTS
    ent_ts::ETS
    target_parameters::TP
    target_states::TST
end

# --- parameter / state selection -------------------------------------------------

function select_actor_parameters(
        ::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SeparateFeatures},
        ps::NamedTuple,
    )
    return (;
        actor_feature_extractor = ps.actor_feature_extractor,
        actor_head = ps.actor_head,
        log_std = ps.log_std,
    )
end

function select_critic_parameters(
        ::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SeparateFeatures},
        ps::NamedTuple,
    )
    return (;
        critic_feature_extractor = ps.critic_feature_extractor,
        critic_head = ps.critic_head,
    )
end

function select_actor_parameters(
        ::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SharedFeatures},
        ps::NamedTuple,
    )
    # Shared encoder is owned by the critic TrainState; actor merges it as Const via data.
    return (;
        actor_head = ps.actor_head,
        log_std = ps.log_std,
    )
end

function select_critic_parameters(
        ::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SharedFeatures},
        ps::NamedTuple,
    )
    return (;
        feature_extractor = ps.feature_extractor,
        critic_head = ps.critic_head,
    )
end

function select_actor_states(
        ::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SeparateFeatures},
        st::NamedTuple,
    )
    return (;
        actor_feature_extractor = st.actor_feature_extractor,
        actor_head = st.actor_head,
    )
end

function select_critic_states(
        ::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SeparateFeatures},
        st::NamedTuple,
    )
    return (;
        critic_feature_extractor = st.critic_feature_extractor,
        critic_head = st.critic_head,
    )
end

function select_actor_states(
        ::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SharedFeatures},
        st::NamedTuple,
    )
    return (; actor_head = st.actor_head)
end

function select_critic_states(
        ::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SharedFeatures},
        st::NamedTuple,
    )
    return (;
        feature_extractor = st.feature_extractor,
        critic_head = st.critic_head,
    )
end

function merge_actor_critic_parameters(actor_ps::NamedTuple, critic_ps::NamedTuple)
    return merge(actor_ps, critic_ps)
end

function merge_actor_critic_states(actor_st::NamedTuple, critic_st::NamedTuple)
    return merge(actor_st, critic_st)
end

function split_states(ts::SACTrainState, st::NamedTuple)
    actor_st = (; (k => st[k] for k in keys(ts.actor_ts.states))...)
    critic_st = (; (k => st[k] for k in keys(ts.critic_ts.states))...)
    return actor_st, critic_st
end

# --- accessors -------------------------------------------------------------------

function parameters(ts::PPOTrainState)
    return ts.ts.parameters
end

function parameters(ts::SACTrainState)
    return merge_actor_critic_parameters(ts.actor_ts.parameters, ts.critic_ts.parameters)
end

function states(ts::PPOTrainState)
    return ts.ts.states
end

function states(ts::SACTrainState)
    return merge_actor_critic_states(ts.actor_ts.states, ts.critic_ts.states)
end

function set_states!(ts::PPOTrainState, st)
    ts.ts = Accessors.@set ts.ts.states = st
    return ts
end

function set_states!(ts::SACTrainState, st)
    actor_st, critic_st = split_states(ts, st)
    ts.actor_ts = Accessors.@set ts.actor_ts.states = actor_st
    ts.critic_ts = Accessors.@set ts.critic_ts.states = critic_st
    return ts
end

function lux_train_state(ts::PPOTrainState)
    return ts.ts
end

function set_lux_train_state!(ts::PPOTrainState, inner)
    ts.ts = inner
    return ts
end

function entropy_parameters(ts::SACTrainState)
    return ts.ent_ts.parameters
end

function entropy_coefficient(ts::SACTrainState)
    return exp(first(ts.ent_ts.parameters.log_ent_coef))
end

function Optimisers.adjust!(ts::PPOTrainState, eta::Real)
    Optimisers.adjust!(ts.ts, eta)
    return ts
end

function Optimisers.adjust!(ts::SACTrainState, eta::Real)
    Optimisers.adjust!(ts.actor_ts, eta)
    Optimisers.adjust!(ts.critic_ts, eta)
    Optimisers.adjust!(ts.ent_ts, eta)
    return ts
end
