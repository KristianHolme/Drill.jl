# Selecting the actor and critic parts of a model's parameters and states

function select_actor_parameters(
        ::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SeparateFeatures},
        ps::NamedTuple,
    )
    return (;
        actor_feature_extractor = ps.actor_feature_extractor,
        actor_head = ps.actor_head,
        log_std = ps.log_std,
    )
end

function select_critic_parameters(
        ::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SeparateFeatures},
        ps::NamedTuple,
    )
    return (;
        critic_feature_extractor = ps.critic_feature_extractor,
        critic_head = ps.critic_head,
    )
end

function select_actor_parameters(
        ::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SharedFeatures},
        ps::NamedTuple,
    )
    # The shared encoder is trained with the critic; the actor loss uses it as a constant.
    return (;
        actor_head = ps.actor_head,
        log_std = ps.log_std,
    )
end

function select_critic_parameters(
        ::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SharedFeatures},
        ps::NamedTuple,
    )
    return (;
        feature_extractor = ps.feature_extractor,
        critic_head = ps.critic_head,
    )
end

function select_actor_states(
        ::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SeparateFeatures},
        st::NamedTuple,
    )
    return (;
        actor_feature_extractor = st.actor_feature_extractor,
        actor_head = st.actor_head,
    )
end

function select_critic_states(
        ::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SeparateFeatures},
        st::NamedTuple,
    )
    return (;
        critic_feature_extractor = st.critic_feature_extractor,
        critic_head = st.critic_head,
    )
end

function select_actor_states(
        ::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SharedFeatures},
        st::NamedTuple,
    )
    return (; actor_head = st.actor_head)
end

function select_critic_states(
        ::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SharedFeatures},
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

"""
Project `src` onto the keys of `template` without generators (Zygote-friendly).
"""
function project_namedtuple(src::NamedTuple, ::NamedTuple{names}) where {names}
    return NamedTuple{names}(map(n -> src[n], names))
end
