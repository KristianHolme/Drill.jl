# Learner state and update functions.
#
# An update takes the algorithm, the model, a learner state and a batch, and returns the
# updated learner state plus a NamedTuple of scalar metrics. It does not touch RLCache or
# any global state, so the same function runs eagerly on the CPU or compiled by a backend.
# Like `Optimisers.update!`, an update may reuse the memory of the learner it is given:
# callers use only the returned learner afterwards.

"""
    value_and_gradient(ad, f, x, args...) -> (loss, aux, grad)

Differentiate `f(x, args...) -> (loss, aux)` with respect to `x` using the AD backend
`ad`; `args` are constants. `aux` is returned unchanged and is not differentiated. Methods
live in package extensions: load Zygote for `AutoZygote()` or Enzyme for `AutoEnzyme()`.
"""
function value_and_gradient(ad, f, x, args...)
    throw(
        ArgumentError(
            "No gradient method for $(typeof(ad)). Load the AD package first, e.g. `using Zygote` for AutoZygote() or `using Enzyme` for AutoEnzyme().",
        ),
    )
end

_array_leaves(x) = filter(l -> l isa AbstractArray, fleaves(x))

"""
    global_norm(grads)

The L2 norm of all arrays in a nested gradient structure.
"""
function global_norm(grads)
    leaves = _array_leaves(grads)
    isempty(leaves) && return 0.0f0
    return sqrt(sum(l -> sum(abs2, l), leaves))
end

"""
    clip_by_global_norm!(grads, max_norm, norm) -> grads

Scale `grads` in place so that their global norm is at most `max_norm`, as in PyTorch's
`clip_grad_norm_`.
"""
function clip_by_global_norm!(grads, max_norm, norm)
    scale = min(one(norm), max_norm / (norm + oftype(norm, 1.0e-6)))
    foreach(g -> g .*= scale, _array_leaves(grads))
    return grads
end

"""
    AbstractLearner

Immutable training state of an algorithm: parameters, model states and optimizer states.
Use [`parameters`](@ref) and [`states`](@ref) for the full model parameters and states.
"""
abstract type AbstractLearner end

# ------------------------------------------------------------
# PPO
# ------------------------------------------------------------

"""
    PPOLearner

Learner state for PPO: model parameters `ps`, model states `st` and optimizer state.
"""
struct PPOLearner{P, S, O} <: AbstractLearner
    ps::P
    st::S
    opt_state::O
end

"""
    prepare_optimizer(device, rule)

The optimizer rule to use for parameters on `device`. Compiled backends may need their
hyperparameters stored as device numbers; the default returns `rule` unchanged.
"""
prepare_optimizer(device, rule) = rule

"""
    init_learner(alg, model, ps, st; rng, device = identity) -> AbstractLearner

The initial learner state for `alg`, with optimizer states set up for `ps`. `ps` and `st`
must already be on `device`, which is used for any other arrays the learner creates.
"""
function init_learner(alg::PPO, model, ps, st; rng::AbstractRNG = default_rng(), device = identity)
    rule = prepare_optimizer(device, make_optimizer(alg))
    return PPOLearner(ps, st, Optimisers.setup(rule, ps))
end

parameters(learner::PPOLearner) = learner.ps
states(learner::PPOLearner) = learner.st
with_states(learner::PPOLearner, st) = PPOLearner(learner.ps, st, learner.opt_state)

function set_learning_rate(learner::PPOLearner, eta::Real)
    opt_state = deepcopy(learner.opt_state)
    Optimisers.adjust!(opt_state, eta)
    return PPOLearner(learner.ps, learner.st, opt_state)
end

"""
    ppo_update(alg::PPO, model, learner::PPOLearner, batch, ad) -> (learner, metrics)

One PPO gradient step on a minibatch. The returned learner may share memory with
`learner`, which must not be used afterwards ; `batch = (observations, actions, advantages,
returns, old_logprobs, old_values)`: advantage normalization, loss, gradient, global norm
clipping and optimizer step.
"""
function ppo_update(alg::PPO, model, learner::PPOLearner, batch, ad)
    observations, actions, advantages, returns, old_logprobs, old_values = batch
    data = (
        observations, actions, maybe_normalize(advantages, alg.advantage_strategy),
        returns, old_logprobs, old_values,
    )
    loss, (new_st, stats), grads = value_and_gradient(ad, _ppo_loss, learner.ps, alg, model, learner.st, data)
    grad_norm = global_norm(grads)
    clip_by_global_norm!(grads, alg.max_grad_norm, grad_norm)
    opt_state, ps = Optimisers.update!(learner.opt_state, learner.ps, grads)
    metrics = (;
        loss,
        policy_loss = stats.policy_loss,
        value_loss = stats.value_loss,
        entropy_loss = stats.entropy_loss,
        approx_kl_div = stats.approx_kl_div,
        clip_fraction = stats.clip_fraction,
        grad_norm,
    )
    return PPOLearner(ps, new_st, opt_state), metrics
end

function _ppo_loss(ps, alg, model, st, data)
    loss, new_st, stats = alg(model, ps, st, data)
    return loss, (new_st, stats)
end

# ------------------------------------------------------------
# SAC
# ------------------------------------------------------------

"""
    SACLearner

Learner state for SAC: actor and critic parameters and states, target critic parameters
and states, the log entropy coefficient, their optimizer states, and the RNG used for
action sampling during updates.
"""
struct SACLearner{AP, AS, CP, CS, TP, TS, E, AO, CO, EO, R} <: AbstractLearner
    actor_ps::AP
    actor_st::AS
    critic_ps::CP
    critic_st::CS
    target_ps::TP
    target_st::TS
    log_ent_coef::E
    actor_opt::AO
    critic_opt::CO
    ent_opt::EO
    rng::R
end

# Device transfer (fmap) moves the arrays and optimizer states but keeps the RNG object.
function Functors.functor(::Type{<:SACLearner}, l)
    children = (;
        l.actor_ps, l.actor_st, l.critic_ps, l.critic_st, l.target_ps, l.target_st,
        l.log_ent_coef, l.actor_opt, l.critic_opt, l.ent_opt,
    )
    rebuild = c -> SACLearner(
        c.actor_ps, c.actor_st, c.critic_ps, c.critic_st, c.target_ps, c.target_st,
        c.log_ent_coef, c.actor_opt, c.critic_opt, c.ent_opt, l.rng,
    )
    return children, rebuild
end

function init_learner(alg::SAC, model, ps, st; rng::AbstractRNG = default_rng(), device = identity)
    actor_ps = select_actor_parameters(model, ps)
    critic_ps = select_critic_parameters(model, ps)
    log_ent_coef = device(init_entropy_coefficient(alg.ent_coef))
    opt = prepare_optimizer(device, make_optimizer(alg))
    return SACLearner(
        actor_ps,
        select_actor_states(model, st),
        critic_ps,
        select_critic_states(model, st),
        fmap(copy, critic_ps),
        select_critic_states(model, st),
        log_ent_coef,
        Optimisers.setup(opt, actor_ps),
        Optimisers.setup(opt, critic_ps),
        Optimisers.setup(opt, log_ent_coef),
        rng,
    )
end

parameters(learner::SACLearner) = merge_actor_critic_parameters(learner.actor_ps, learner.critic_ps)
states(learner::SACLearner) = merge_actor_critic_states(learner.actor_st, learner.critic_st)

function with_states(learner::SACLearner, st)
    actor_st = project_namedtuple(st, learner.actor_st)
    critic_st = project_namedtuple(st, learner.critic_st)
    return SACLearner(
        learner.actor_ps, actor_st, learner.critic_ps, critic_st, learner.target_ps,
        learner.target_st, learner.log_ent_coef, learner.actor_opt, learner.critic_opt,
        learner.ent_opt, learner.rng,
    )
end

# `sum` over the one-element array instead of `first`, which compiled backends cannot trace
entropy_coefficient(learner::SACLearner) = exp(sum(learner.log_ent_coef.log_ent_coef))

function set_learning_rate(learner::SACLearner, eta::Real)
    actor_opt, critic_opt, ent_opt = deepcopy((learner.actor_opt, learner.critic_opt, learner.ent_opt))
    Optimisers.adjust!(actor_opt, eta)
    Optimisers.adjust!(critic_opt, eta)
    Optimisers.adjust!(ent_opt, eta)
    return SACLearner(
        learner.actor_ps, learner.actor_st, learner.critic_ps, learner.critic_st,
        learner.target_ps, learner.target_st, learner.log_ent_coef, actor_opt, critic_opt,
        ent_opt, learner.rng,
    )
end

_learns_entropy(::AutoEntropyCoefficient) = true
_learns_entropy(::FixedEntropyCoefficient) = false

"""
    polyak!(target, source, tau) -> target

Move every array in `target` towards the matching array in `source`, in place:
`target = (1 - tau) * target + tau * source`.
"""
function polyak!(target, source, tau)
    foreach(_array_leaves(target), _array_leaves(source)) do t, s
        t .= (1 - tau) .* t .+ tau .* s
    end
    return target
end

"""
    sac_update(alg::SAC, model, learner::SACLearner, batch, ad, target_entropy, update_target::Val)
        -> (learner, metrics)

One SAC gradient step. The returned learner may share memory with `learner`, which must
not be used afterwards. The step computes target Q values, critic step, actor step, entropy coefficient
step (from the actor's log-probabilities), and a Polyak update of the target critic when
`update_target` is `Val(true)`. The entropy coefficient used for the target and the actor
loss is the one from before this step.
"""
function sac_update(alg::SAC, model, learner::SACLearner, batch, ad, target_entropy, ::Val{update_target}) where {update_target}
    rng = learner.rng
    ent_coef = entropy_coefficient(learner)
    full_ps = parameters(learner)
    full_st = states(learner)

    target_q_values = compute_target_q_values(
        alg,
        model,
        full_ps,
        full_st,
        (
            next_observations = batch.next_observations,
            terminated = batch.terminated,
            log_ent_coef = learner.log_ent_coef,
            rewards = batch.rewards,
            target_ps = learner.target_ps,
            target_st = learner.target_st,
        );
        rng,
    )

    critic_data = (; batch.observations, batch.actions, target_q_values)
    critic_loss, (critic_st, critic_stats), critic_grad = value_and_gradient(
        ad, _sac_critic_loss, learner.critic_ps,
        alg, model, learner.actor_ps, learner.actor_st, learner.critic_st, critic_data, rng,
    )
    critic_grad_norm = global_norm(critic_grad)
    critic_opt, critic_ps = Optimisers.update!(learner.critic_opt, learner.critic_ps, critic_grad)

    actor_data = (; batch.observations, ent_coef)
    actor_loss, (actor_st, log_probs), actor_grad = value_and_gradient(
        ad, _sac_actor_loss, learner.actor_ps,
        alg, model, learner.actor_st, critic_ps, critic_st, actor_data, rng,
    )
    actor_grad_norm = global_norm(actor_grad)
    actor_opt, actor_ps = Optimisers.update!(learner.actor_opt, learner.actor_ps, actor_grad)

    log_ent_coef, ent_opt, entropy_loss = _entropy_step(alg.ent_coef, learner, log_probs, target_entropy, ad)

    target_ps, target_st = if update_target
        polyak!(learner.target_ps, critic_ps, alg.tau), critic_st
    else
        learner.target_ps, learner.target_st
    end

    new_learner = SACLearner(
        actor_ps, actor_st, critic_ps, critic_st, target_ps, target_st, log_ent_coef,
        actor_opt, critic_opt, ent_opt, rng,
    )
    metrics = (;
        actor_loss,
        critic_loss,
        entropy_loss,
        mean_q_values = critic_stats.mean_q_values,
        entropy_coefficient = ent_coef,
        grad_norm = sqrt(critic_grad_norm^2 + actor_grad_norm^2),
    )
    return new_learner, metrics
end

function _sac_critic_loss(critic_ps, alg, model, actor_ps, actor_st, critic_st, data, rng)
    loss, new_st, stats = sac_critic_loss(
        alg, model,
        merge_actor_critic_parameters(actor_ps, critic_ps),
        merge_actor_critic_states(actor_st, critic_st),
        data; rng,
    )
    return loss, (project_namedtuple(new_st, critic_st), stats)
end

function _sac_actor_loss(actor_ps, alg, model, actor_st, critic_ps, critic_st, data, rng)
    loss, new_st, log_probs = sac_actor_loss(
        alg, model,
        merge_actor_critic_parameters(actor_ps, critic_ps),
        merge_actor_critic_states(actor_st, critic_st),
        data; rng,
    )
    return loss, (project_namedtuple(new_st, actor_st), log_probs)
end

_entropy_loss(p, c) = (-(sum(p.log_ent_coef) * c), nothing)

function _entropy_step(ent::AbstractEntropyCoefficient, learner::SACLearner, log_probs, target_entropy, ad)
    if !_learns_entropy(ent)
        return learner.log_ent_coef, learner.ent_opt, zero(eltype(log_probs))
    end
    c = mean(log_probs .+ target_entropy)
    loss, _, grad = value_and_gradient(ad, _entropy_loss, learner.log_ent_coef, c)
    ent_opt, log_ent_coef = Optimisers.update!(learner.ent_opt, learner.log_ent_coef, grad)
    return log_ent_coef, ent_opt, loss
end
