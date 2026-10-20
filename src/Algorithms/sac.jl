@kwdef struct SAC{T <: AbstractFloat, E <: AbstractEntropyCoefficient, LR} <: OffPolicyAlgorithm
    learning_rate::LR = 3.0f-4
    buffer_capacity::Int = 1_000_000
    start_steps::Int = 100
    batch_size::Int = 256
    tau::T = 0.005f0
    gamma::T = 0.99f0
    train_freq::Int = 1
    gradient_steps::Int = 1
    ent_coef::E = AutoEntropyCoefficient()
    target_update_interval::Int = 1
    optimizer::Type{<:Optimisers.AbstractRule} = Optimisers.Adam
end

function make_optimizer(optimizer_type::Type{<:Optimisers.Adam}, alg::SAC)
    return optimizer_type(eta = learning_rate(alg, 0), epsilon = 1.0f-5)
end

function make_optimizer(optimizer_type::Type{<:Optimisers.AbstractRule}, alg::SAC)
    return optimizer_type(learning_rate(alg, 0))
end

learning_rate(alg::SAC{T}, progress::Real) where {T} = T(schedule_value(alg.learning_rate, progress))

make_optimizer(alg::SAC) = make_optimizer(alg.optimizer, alg)

action_adapter(::SAC, ::Box) = ScaleAdapter()
has_twin_critics(::SAC) = true
has_target_networks(::SAC) = true
has_entropy_tuning(::SAC) = true
uses_replay(::SAC) = true
critic_type(::SAC) = QCritic()

function get_target_entropy(ent_coef::AutoEntropyCoefficient{T}, action_space) where {T}
    if ent_coef.target isa AutoEntropyTarget
        return -T(prod(size(action_space)))
    elseif ent_coef.target isa FixedEntropyTarget
        return ent_coef.target.target
    end
    error("Unknown entropy target type: $(typeof(ent_coef.target))")
end

get_target_entropy(::FixedEntropyCoefficient, action_space) = nothing

function get_gradient_steps(alg::SAC, train_freq::Int = alg.train_freq, n_envs::Int = 1)
    if alg.gradient_steps == -1
        return train_freq * n_envs
    end
    return alg.gradient_steps
end

function SACModel(
        observation_space::Union{Discrete, Box{T}},
        action_space::Box{T};
        log_std_init::T = T(-3),
        hidden_dims = [512, 512],
        activation = relu,
        shared_features::Bool = false,
        critic_type::CriticType = QCritic(),
    ) where {T}
    return ContinuousActorCriticModel(
        observation_space,
        action_space;
        log_std_init,
        hidden_dims,
        activation,
        shared_features,
        critic_type,
    )
end

function init_entropy_coefficient(entropy_coefficient::FixedEntropyCoefficient)
    return (; log_ent_coef = [entropy_coefficient.coef |> log])
end

function init_entropy_coefficient(entropy_coefficient::AutoEntropyCoefficient)
    return (; log_ent_coef = [entropy_coefficient.initial_value |> log])
end

"""
    sac_actor_loss(alg::SAC, layer, actor_ps, actor_st, critic_ps, critic_st, data; rng) -> (loss, actor_st, log_probs)

The SAC actor loss. Actor and critic parameters are separate arguments, so a gradient with
respect to `actor_ps` treats the critic as a constant without merging the two.
"""
function sac_actor_loss(
        ::SAC,
        layer::ContinuousActorCriticModel{<:Any, <:Any, <:Any, QCritic},
        actor_ps,
        actor_st,
        critic_ps,
        critic_st,
        data;
        rng::AbstractRNG = default_rng(),
    )
    ent_coef = data.ent_coef
    actor_feats, actor_st = _sac_actor_features(layer, data.observations, actor_ps, actor_st, critic_ps, critic_st)
    actions_pi, log_probs_pi, actor_st = action_log_prob_from_features(layer, actor_feats, actor_ps, actor_st; rng)
    critic_feats, critic_st = critic_features(layer, data.observations, critic_ps, critic_st)
    q_values, _ = get_values_from_features(layer, critic_feats, actions_pi, critic_ps, critic_st)
    min_q_values = vec(minimum(q_values, dims = 1))
    loss = mean(ent_coef .* log_probs_pi - min_q_values)
    return loss, actor_st, log_probs_pi
end

# Shared features belong to the critic; the actor uses them as a constant.
function _sac_actor_features(
        layer::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SharedFeatures},
        obs, actor_ps, actor_st, critic_ps, critic_st,
    )
    feats, _ = actor_features(layer, obs, critic_ps, critic_st)
    return feats, actor_st
end

function _sac_actor_features(
        layer::ContinuousActorCriticModel{<:Any, <:Any, <:Any, <:Any, SeparateFeatures},
        obs, actor_ps, actor_st, critic_ps, critic_st,
    )
    return actor_features(layer, obs, actor_ps, actor_st)
end

function compute_target_q_values(
        alg::SAC,
        layer::ContinuousActorCriticModel{<:Any, <:Any, <:Any, QCritic},
        ps,
        st,
        data;
        rng::AbstractRNG = default_rng(),
    )
    next_obs = data.next_observations
    ent_coef = exp(sum(data.log_ent_coef.log_ent_coef))
    next_actions, next_log_probs, st = action_log_prob(layer, next_obs, ps, st; rng)
    ps_with_target = merge(ps, data.target_ps)
    st_with_target = merge(st, data.target_st)
    next_q_vals, _ = predict_values(layer, next_obs, next_actions, ps_with_target, st_with_target)
    min_next_q = vec(minimum(next_q_vals, dims = 1))
    T = eltype(min_next_q)
    next_q_vals_with_entropy = min_next_q .- ent_coef .* next_log_probs
    # ifelse, not multiplication by a 0/1 mask: 0 * NaN is NaN
    next_values = ifelse.(data.terminated, zero(T), next_q_vals_with_entropy)
    target_q_values = data.rewards .+ alg.gamma .* next_values
    return target_q_values
end

"""
    sac_critic_loss(alg::SAC, layer, critic_ps, critic_st, data) -> (loss, critic_st, stats)

The SAC critic loss, from the critic's parameters and states alone.
"""
function sac_critic_loss(
        ::SAC,
        layer::ContinuousActorCriticModel{<:Any, <:Any, <:Any, QCritic},
        critic_ps,
        critic_st,
        data,
    )
    feats, critic_st = critic_features(layer, data.observations, critic_ps, critic_st)
    current_q_values, critic_st = get_values_from_features(layer, feats, data.actions, critic_ps, critic_st)
    T = eltype(current_q_values)
    δ = current_q_values .- reshape(data.target_q_values, 1, :)
    critic_loss = T(0.5) * sum(mean(abs2, δ; dims = 2))
    stats = (mean_q_values = mean(current_q_values),)
    return critic_loss, critic_st, stats
end

function process_action(action, action_space::Box{T}, ::SAC) where {T}
    if eltype(action) != T
        @warn "Action type mismatch: $(eltype(action)) != $T"
        action = convert.(T, action)
    end
    low = action_space.low
    high = action_space.high
    return action .* (high - low) ./ T(2) + (low + high) ./ T(2)
end
