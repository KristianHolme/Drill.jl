@kwdef struct SAC{T <: AbstractFloat, E <: AbstractEntropyCoefficient} <: OffPolicyAlgorithm
    learning_rate::T = 3.0f-4
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
    return optimizer_type(eta = alg.learning_rate, epsilon = 1.0f-5)
end

function make_optimizer(optimizer_type::Type{<:Optimisers.AbstractRule}, alg::SAC)
    return optimizer_type(alg.learning_rate)
end

make_optimizer(alg::SAC) = make_optimizer(alg.optimizer, alg)

action_adapter(::SAC, ::Box) = TanhScaleAdapter()
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

function SACLayer(
        observation_space::Union{Discrete, Box{T}},
        action_space::Box{T};
        log_std_init::T = T(-3),
        hidden_dims = [512, 512],
        activation = relu,
        shared_features::Bool = false,
        critic_type::CriticType = QCritic(),
    ) where {T}
    return ContinuousActorCriticLayer(
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

function sac_actor_loss(
        ::SAC,
        layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, QCritic},
        ps,
        st,
        data;
        rng::AbstractRNG = default_rng(),
    )
    ent_coef = data.ent_coef
    actions_pi, log_probs_pi, st = action_log_prob(layer, data.observations, ps, st; rng)
    q_values, st = predict_values(layer, data.observations, actions_pi, ps, st)
    min_q_values = vec(minimum(q_values, dims = 1))
    loss = mean(ent_coef .* log_probs_pi - min_q_values)
    return loss, st, NamedTuple()
end

function compute_target_q_values(
        alg::SAC,
        layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, QCritic},
        ps,
        st,
        data;
        rng::AbstractRNG = default_rng(),
    )
    next_obs = data.next_observations
    ent_coef = exp(first(data.log_ent_coef.log_ent_coef))
    next_actions, next_log_probs, st = action_log_prob(layer, next_obs, ps, st; rng)
    ps_with_target = merge_params(ps, data.target_ps)
    st_with_target = merge(st, data.target_st)
    next_q_vals, _ = predict_values(layer, next_obs, next_actions, ps_with_target, st_with_target)
    min_next_q = vec(minimum(next_q_vals, dims = 1))
    T = eltype(min_next_q)
    mask = T.(.!data.terminated)
    next_q_vals_with_entropy = min_next_q .- ent_coef .* next_log_probs
    target_q_values = data.rewards .+ alg.gamma .* mask .* next_q_vals_with_entropy
    return target_q_values
end

function sac_critic_loss(
        alg::SAC,
        layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, QCritic},
        ps,
        st,
        data;
        rng::AbstractRNG = default_rng(),
    )
    current_q_values, new_st = predict_values(layer, data.observations, data.actions, ps, st)
    T = eltype(current_q_values)
    δ = current_q_values .- reshape(data.target_q_values, 1, :)
    critic_loss = T(0.5) * sum(mean(abs2, δ; dims = 2))
    stats = (mean_q_values = mean(current_q_values),)
    return critic_loss, new_st, stats
end

function (alg::SAC)(::ContinuousActorCriticLayer, ps, st, batch_data)
    error("SAC algorithm object should not be called directly. Use SAC objectives instead.")
end

struct SACEntropyObjective end

function (::SACEntropyObjective)(model, ps, st, data)
    log_ent_coef = first(ps.log_ent_coef)
    loss = -(log_ent_coef * data.c)
    return loss, st, NamedTuple()
end

struct SACCriticObjective{A, R}
    alg::A
    rng::R
end

function (objective::SACCriticObjective)(model, ps, st, data)
    full_ps = merge_actor_critic_parameters(data.actor_ps, ps)
    full_st = merge_actor_critic_states(data.actor_st, st)
    loss, new_full_st, stats = sac_critic_loss(
        objective.alg,
        model,
        full_ps,
        full_st,
        data;
        rng = objective.rng,
    )
    new_st = project_namedtuple(new_full_st, st)
    return loss, new_st, stats
end

struct SACActorObjective{A, R}
    alg::A
    rng::R
end

function (objective::SACActorObjective)(model, ps, st, data)
    full_ps = merge_actor_critic_parameters(ps, data.critic_ps)
    full_st = merge_actor_critic_states(st, data.critic_st)
    loss, new_full_st, stats = sac_actor_loss(
        objective.alg,
        model,
        full_ps,
        full_st,
        data;
        rng = objective.rng,
    )
    new_st = project_namedtuple(new_full_st, st)
    return loss, new_st, stats
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
