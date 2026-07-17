abstract type AbstractAdvantageStrategy end
struct NormalizeAdvantages <: AbstractAdvantageStrategy end
struct RawAdvantages <: AbstractAdvantageStrategy end

abstract type AbstractClipVFStrategy{T <: AbstractFloat} end
struct ClipVF{T <: AbstractFloat} <: AbstractClipVFStrategy{T}
    value::T
end
struct NoClipVF{T <: AbstractFloat} <: AbstractClipVFStrategy{T} end

abstract type AbstractKLTargetStrategy{T <: AbstractFloat} end
struct KLTarget{T <: AbstractFloat} <: AbstractKLTargetStrategy{T}
    value::T
end
struct NoKLTarget{T <: AbstractFloat} <: AbstractKLTargetStrategy{T} end

struct PPO{
        T <: AbstractFloat,
        AS <: AbstractAdvantageStrategy,
        CVS <: AbstractClipVFStrategy{T},
        KLS <: AbstractKLTargetStrategy{T},
    } <: OnPolicyAlgorithm
    gamma::T
    gae_lambda::T
    clip_range::T
    ent_coef::T
    vf_coef::T
    max_grad_norm::T
    advantage_strategy::AS
    clip_vf_strategy::CVS
    target_kl_strategy::KLS
    n_steps::Int
    batch_size::Int
    epochs::Int
    learning_rate::T
    optimizer::Type{<:Optimisers.AbstractRule}
end

function Base.convert(::Type{AbstractAdvantageStrategy}, x::Bool)
    if x
        return NormalizeAdvantages()
    end
    return RawAdvantages()
end

function Base.convert(::Type{AbstractClipVFStrategy{T}}, ::Nothing) where {T <: AbstractFloat}
    return NoClipVF{T}()
end

function Base.convert(::Type{AbstractClipVFStrategy{T}}, x::Real) where {T <: AbstractFloat}
    return ClipVF{T}(T(x))
end

function Base.convert(::Type{AbstractKLTargetStrategy{T}}, ::Nothing) where {T <: AbstractFloat}
    return NoKLTarget{T}()
end

function Base.convert(::Type{AbstractKLTargetStrategy{T}}, x::Real) where {T <: AbstractFloat}
    return KLTarget{T}(T(x))
end

normalize_advantage(::PPO{<:Any, NormalizeAdvantages}) = true
normalize_advantage(::PPO{<:Any, RawAdvantages}) = false

clip_range_vf(::PPO{<:Any, <:Any, <:NoClipVF}) = nothing
function clip_range_vf(alg::PPO{<:Any, <:Any, <:ClipVF})
    return alg.clip_vf_strategy.value
end

target_kl(::PPO{<:Any, <:Any, <:Any, <:NoKLTarget}) = nothing
function target_kl(alg::PPO{<:Any, <:Any, <:Any, <:KLTarget})
    return alg.target_kl_strategy.value
end

function PPO(;
        gamma::Real = 0.99f0,
        gae_lambda::Real = 0.95f0,
        clip_range::Real = 0.2f0,
        clip_range_vf = nothing,
        ent_coef::Real = 0.0f0,
        vf_coef::Real = 0.5f0,
        max_grad_norm::Real = 0.5f0,
        target_kl = nothing,
        normalize_advantage::Bool = true,
        n_steps::Int = 2048,
        batch_size::Int = 64,
        epochs::Int = 10,
        learning_rate::Real = 3.0f-4,
        optimizer::Type{<:Optimisers.AbstractRule} = Optimisers.Adam,
    )
    T = promote_type(
        typeof(float(gamma)),
        typeof(float(gae_lambda)),
        typeof(float(clip_range)),
        typeof(float(ent_coef)),
        typeof(float(vf_coef)),
        typeof(float(max_grad_norm)),
        typeof(float(learning_rate)),
        isnothing(clip_range_vf) ? Float32 : typeof(float(clip_range_vf)),
        isnothing(target_kl) ? Float32 : typeof(float(target_kl)),
    )
    advantage_strategy = convert(AbstractAdvantageStrategy, normalize_advantage)
    clip_vf_strategy = convert(AbstractClipVFStrategy{T}, clip_range_vf)
    target_kl_strategy = convert(AbstractKLTargetStrategy{T}, target_kl)
    return PPO{T, typeof(advantage_strategy), typeof(clip_vf_strategy), typeof(target_kl_strategy)}(
        T(gamma),
        T(gae_lambda),
        T(clip_range),
        T(ent_coef),
        T(vf_coef),
        T(max_grad_norm),
        advantage_strategy,
        clip_vf_strategy,
        target_kl_strategy,
        n_steps,
        batch_size,
        epochs,
        T(learning_rate),
        optimizer,
    )
end

function make_optimizer(optimizer_type::Type{<:Optimisers.Adam}, alg::PPO)
    return optimizer_type(eta = alg.learning_rate, epsilon = 1.0f-5)
end

function make_optimizer(optimizer_type::Type{<:Optimisers.AbstractRule}, alg::PPO)
    return optimizer_type(alg.learning_rate)
end

make_optimizer(alg::PPO) = make_optimizer(alg.optimizer, alg)

action_adapter(::PPO, ::Discrete) = DiscreteAdapter()
action_adapter(::PPO, ::Box) = ClampAdapter()
has_twin_critics(::PPO) = false
has_target_networks(::PPO) = false
has_entropy_tuning(::PPO) = false
uses_replay(::PPO) = false
critic_type(::PPO) = VCritic()

function normalize(advantages::AbstractVector{T}) where {T}
    mean_adv = mean(advantages)
    std_adv = std(advantages)
    epsilon = T(1.0e-8)
    norm_advantages = (advantages .- mean_adv) ./ (std_adv + epsilon)
    return norm_advantages
end

function clip_range!(values, old_values, clip_range)
    for i in eachindex(values)
        diff = values[i] - old_values[i]
        clipped_diff = clamp(diff, -clip_range, clip_range)
        values[i] = old_values[i] + clipped_diff
    end
    return nothing
end

function clip_range(old_values, values, clip_range)
    return old_values .+ clamp(values .- old_values, -clip_range, clip_range)
end


#TODO: vectorize this?
function normalize!(values::AbstractVector{T}) where {T}
    mean_values = mean(values)
    std_values = std(values)
    epsilon = T(1.0e-8)
    values .= (values .- mean_values) ./ (std_values + epsilon)
    return nothing
end

function maybe_normalize!(advantages::AbstractVector{T}, ::NormalizeAdvantages) where {T}
    normalize!(advantages)
    return advantages
end
function maybe_normalize!(advantages::AbstractVector{T}, ::RawAdvantages) where {T}
    return advantages
end

function maybe_normalize_batch_data(batch_data, strategy::AbstractAdvantageStrategy)
    advantages = maybe_normalize!(copy(batch_data[3]), strategy)
    return (
        batch_data[1],
        batch_data[2],
        advantages,
        batch_data[4],
        batch_data[5],
        batch_data[6],
    )
end

function maybe_clip_range(old_values, values, ::NoClipVF)
    return values
end
function maybe_clip_range(old_values, values, strategy::ClipVF)
    return clip_range(old_values, values, strategy.value)
end

function (alg::PPO{T})(layer::AbstractActorCriticLayer, ps, st, batch_data) where {T}
    observations = batch_data[1]
    actions = batch_data[2]
    advantages = batch_data[3]
    returns = batch_data[4]
    old_logprobs = batch_data[5]
    old_values = batch_data[6]

    values, log_probs, entropy, st = evaluate_actions(layer, observations, actions, ps, st)
    values = maybe_clip_range(old_values, values, alg.clip_vf_strategy)

    r = exp.(log_probs - old_logprobs)
    ratio_clipped = clamp.(r, 1 - alg.clip_range, 1 + alg.clip_range)
    p_loss = -mean(min.(r .* advantages, ratio_clipped .* advantages))
    ent_loss = -mean(entropy)

    v_loss = mean((values .- returns) .^ 2)
    loss = p_loss + alg.ent_coef * ent_loss + alg.vf_coef * v_loss

    # Calculate statistics
    clip_fraction = mean(r .!= ratio_clipped)
    #approx kl div
    log_ratio = log_probs - old_logprobs
    approx_kl_div = mean(exp.(log_ratio) .- 1 .- log_ratio)

    stats = (;
        policy_loss = p_loss,
        value_loss = v_loss,
        entropy_loss = ent_loss,
        clip_fraction = clip_fraction,
        approx_kl_div = approx_kl_div,
        entropy = mean(entropy),
        ratio = mean(r),
    )

    return loss, st, stats
end


# Helper function to process actions: ensure correct type and clipping for Box
#TODO performance
function process_action(action, action_space::Box{T}, ::PPO) where {T}
    # First check if type conversion is needed
    if eltype(action) != T
        @warn "Action type mismatch: $(eltype(action)) != $T"
        action = convert.(T, action)
    end
    # Then clip to bounds element-wise
    action = clamp.(action, action_space.low, action_space.high)
    return action
end

# Helper function to process actions: validate integer discrete actions
function process_action(action::Integer, action_space::Discrete, ::PPO)
    @assert action in action_space "Action $(action) is out of bounds for Discrete($(action_space.n), $(action_space.start))"
    return action
end

function prepare_training_actions(actions::AbstractArray, action_space::Box)
    return actions
end

function prepare_training_actions(actions::AbstractArray{<:Integer}, action_space::Discrete)
    return discrete_to_onehotbatch(actions, action_space)
end
