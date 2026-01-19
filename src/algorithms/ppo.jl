"""
    PPO{T <: AbstractFloat} <: OnPolicyAlgorithm

Proximal Policy Optimization algorithm.

# Fields
- `gamma`: Discount factor (default: 0.99)
- `gae_lambda`: GAE lambda for advantage estimation (default: 0.95)
- `clip_range`: PPO clipping parameter (default: 0.2)
- `ent_coef`: Entropy coefficient (default: 0.0)
- `vf_coef`: Value function coefficient (default: 0.5)
- `max_grad_norm`: Maximum gradient norm for clipping (default: 0.5)
- `n_steps`: Steps per rollout before update (default: 2048)
- `batch_size`: Minibatch size (default: 64)
- `epochs`: Number of epochs per update (default: 10)
- `learning_rate`: Optimizer learning rate (default: 3e-4)

# Example
```julia
ppo = PPO(gamma=0.99f0, n_steps=2048, epochs=10)
agent = Agent(model, ppo)
train!(agent, env, ppo, 100_000)
```
"""
@kwdef struct PPO{T <: AbstractFloat} <: OnPolicyAlgorithm
    gamma::T = 0.99f0
    gae_lambda::T = 0.95f0
    clip_range::T = 0.2f0
    clip_range_vf::Union{T, Nothing} = nothing
    ent_coef::T = 0.0f0
    vf_coef::T = 0.5f0
    max_grad_norm::T = 0.5f0
    target_kl::Union{T, Nothing} = nothing
    normalize_advantage::Bool = true
    # Agent parameters moved from ActorCriticAgent
    n_steps::Int = 2048
    batch_size::Int = 64
    epochs::Int = 10
    learning_rate::T = 3.0f-4
end

function Agent(
        layer::AbstractActorCriticLayer, alg::PPO;
        optimizer_type::Type{<:Optimisers.AbstractRule} = Optimisers.Adam,
        stats_window::Int = 100, #TODO not used
        verbose::Int = 1,
        logger = NoTrainingLogger(),
        rng::AbstractRNG = Random.default_rng()
    )

    optimizer = make_optimizer(optimizer_type, alg)
    ps, st = Lux.setup(rng, layer)
    train_state = Lux.Training.TrainState(layer, ps, st, optimizer)
    adapter = action_adapter(alg, action_space(layer))


    logger = convert(AbstractTrainingLogger, logger)
    return Agent(
        layer, alg, adapter, train_state, optimizer_type, stats_window,
        logger, verbose, rng, AgentStats(0, 0), NoAux()
    )
end

function make_optimizer(optimizer_type::Type{<:Optimisers.Adam}, alg::PPO)
    return optimizer_type(eta = alg.learning_rate, epsilon = 1.0f-5)
end

# Traits and adapter selection
action_adapter(::PPO, ::Discrete) = DiscreteAdapter()
action_adapter(::PPO, ::Box) = ClampAdapter()
has_twin_critics(::PPO) = false
has_target_networks(::PPO) = false
has_entropy_tuning(::PPO) = false
uses_replay(::PPO) = false
critic_type(::PPO) = VCritic()

function load_policy_params_and_state!(
        agent::Agent{<:AbstractActorCriticLayer, <:PPO, <:AbstractActionAdapter, <:AbstractRNG, <:AbstractTrainingLogger, <:Any},
        alg::PPO,
        path::AbstractString;
        suffix::String = ".jld2"
    )
    file_path = endswith(path, suffix) ? path : path * suffix
    @info "Loading policy, parameters, and state from $file_path"
    data = load(file_path)
    new_layer = data["layer"]
    new_parameters = data["parameters"]
    new_states = data["states"]
    new_optimizer = make_optimizer(agent.optimizer_type, alg)
    new_train_state = Lux.Training.TrainState(new_layer, new_parameters, new_states, new_optimizer)
    agent.layer = new_layer
    agent.train_state = new_train_state
    return agent
end

#TODO make parameters n_steps, batch_size, epochs, max_steps kwargs, default to values from agent
#TODO refactor, separate out learnig loop and logging


function train!(
        agent::Agent{<:AbstractActorCriticLayer, <:PPO, <:AbstractActionAdapter, <:AbstractRNG, <:AbstractTrainingLogger, <:Any},
        env::AbstractParallelEnv,
        alg::PPO{T}, #TODO remove alg from here, use agent.alg instead
        max_steps::Int;
        ad_type::Lux.Training.AbstractADType = AutoZygote(),
        callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing
    ) where {T}
    to = TimerOutput()
    setup_section = begin_timed_section!(to, "setup")
    n_steps = alg.n_steps
    n_envs = number_of_envs(env)
    roll_buffer = RolloutBuffer(
        observation_space(env), action_space(env),
        alg.gae_lambda, alg.gamma, n_steps, n_envs
    )

    iterations = max_steps ÷ (n_steps * n_envs)
    if iterations == 0
        @warn "max_steps is less than n_steps * n_envs, max_steps: $max_steps, n_steps: $n_steps, n_envs: $n_envs. There will be no training."
    end
    total_steps = iterations * n_steps * n_envs

    agent.verbose > 0 && @info "Training with total_steps: $total_steps,
    iterations: $iterations, n_steps: $n_steps, n_envs: $n_envs"

    progress_meter = Progress(
        total_steps, desc = "Training...",
        showspeed = true, enabled = agent.verbose > 0
    )

    train_state = agent.train_state

    total_entropy_losses = Float32[]
    learning_rates = Float32[]
    total_policy_losses = Float32[]
    total_value_losses = Float32[]
    total_approx_kl_divs = Float32[]
    total_clip_fractions = Float32[]
    total_losses = Float32[]
    total_explained_variances = Float32[]
    total_fps = Float32[]
    total_grad_norms = Float32[]
    end_timed_section!(to, setup_section)

    if !isnothing(callbacks)
        @timeit to "callback: training_start" begin
            if !all(c -> on_training_start(c, Base.@locals), callbacks)
                @warn "Training stopped due to callback failure"
                return nothing
            end
        end
    end

    @timeit to "training_loop" for i in 1:iterations
        learning_rate = alg.learning_rate
        Optimisers.adjust!(agent.train_state, learning_rate)
        push!(learning_rates, learning_rate)

        if !isnothing(callbacks)
            @timeit to "callback: rollout_start" begin
                if !all(c -> on_rollout_start(c, Base.@locals), callbacks)
                    @warn "Training stopped due to callback failure"
                    return nothing
                end
            end
        end
        fps, success = @timeit to "collect_rollout" collect_rollout!(roll_buffer, agent, alg, env; callbacks = callbacks)
        if !success
            @warn "Training stopped due to callback failure"
            return nothing
        end
        push!(total_fps, fps)
        add_step!(agent, n_steps * n_envs)

        increment_step!(agent.logger, n_steps * n_envs)
        log_scalar!(agent.logger, "env/fps", fps)
        log_stats(env, agent.logger)

        if !isnothing(callbacks)
            @timeit to "callback: rollout_end" begin
                if !all(c -> on_rollout_end(c, Base.@locals), callbacks)
                    @warn "Training stopped due to callback failure"
                    return nothing
                end
            end
        end

        data_loader = DataLoader(
            (
                roll_buffer.observations, roll_buffer.actions,
                roll_buffer.advantages, roll_buffer.returns,
                roll_buffer.logprobs, roll_buffer.values,
            ),
            batchsize = alg.batch_size, shuffle = true, parallel = true, rng = agent.rng
        )
        continue_training = true
        entropy_losses = Float32[]
        entropy = Float32[]
        policy_losses = Float32[]
        value_losses = Float32[]
        losses = Float32[]
        approx_kl_divs = Float32[]
        clip_fractions = Float32[]
        grad_norms = Float32[]
        @timeit to "epoch loop" for epoch in 1:alg.epochs
            @timeit to "batch loop" for (i_batch, batch_data) in enumerate(data_loader)
                grads, loss_val, stats, train_state = @timeit to "compute_gradients" Lux.Training.compute_gradients(ad_type, alg, batch_data, train_state)

                if epoch == 1 && i_batch == 1
                    mean_ratio = stats["ratio"]
                    isapprox(mean_ratio - one(mean_ratio), zero(mean_ratio), atol = eps(typeof(mean_ratio))) || @warn "ratios is not 1.0, iter $i, epoch $epoch, batch $i_batch, $mean_ratio"
                end
                @assert !nested_has_nan(grads) "gradient contains nan, iter $i, epoch $epoch, batch $i_batch"
                @assert !nested_has_inf(grads) "gradient not finite, iter $i, epoch $epoch, batch $i_batch"

                current_grad_norm = nested_norm(grads, T)
                # @info "actor grad norm: $(nested_norm(grads.actor_head, T))"
                if grads isa NamedTuple && haskey(grads, :actor_head) && nested_norm(grads.actor_head, T) < 1.0e-3
                    @info "actor grad norm is less than 1.0e-3, iter $i, epoch $epoch, batch $i_batch, $(nested_norm(grads.actor_head, T))"
                end
                # @info "critic grad norm: $(nested_norm(grads.critic_head, T))"
                # @info "log_std grad norm: $(nested_norm(grads.log_std, T))"
                push!(grad_norms, current_grad_norm)

                if !isnothing(alg.max_grad_norm) && current_grad_norm > alg.max_grad_norm
                    nested_scale!(grads, alg.max_grad_norm, current_grad_norm)
                    clipped_grads_norm = nested_norm(grads, T)
                    @assert clipped_grads_norm < alg.max_grad_norm ||
                        clipped_grads_norm ≈ alg.max_grad_norm "gradient norm
                            ($(clipped_grads_norm)) is greater than
                            max_grad_norm ($(alg.max_grad_norm)), iter $i, epoch $epoch, batch $i_batch"
                end
                # @info grads
                # KL divergence check
                if !isnothing(alg.target_kl) && stats["approx_kl_div"] > T(1.5) * alg.target_kl
                    continue_training = false
                    break
                end
                @timeit to "apply_gradients" Lux.Training.apply_gradients!(train_state, grads)

                add_gradient_update!(agent)
                push!(entropy, stats["entropy"])
                push!(entropy_losses, stats["entropy_loss"])
                push!(policy_losses, stats["policy_loss"])
                push!(value_losses, stats["value_loss"])
                push!(approx_kl_divs, stats["approx_kl_div"])
                push!(clip_fractions, stats["clip_fraction"])
                push!(losses, loss_val)
            end
            if !continue_training
                @info "Early stopping at epoch $epoch in iteration $i, due to KL divergence"
                break
            end
        end

        explained_variance = 1 - var(roll_buffer.values .- roll_buffer.returns) / var(roll_buffer.returns)
        push!(total_explained_variances, explained_variance)
        push!(total_entropy_losses, mean(entropy_losses))
        push!(total_policy_losses, mean(policy_losses))
        push!(total_value_losses, mean(value_losses))
        push!(total_approx_kl_divs, mean(approx_kl_divs))
        push!(total_clip_fractions, mean(clip_fractions))
        push!(total_losses, mean(losses))
        push!(total_grad_norms, mean(grad_norms))
        if agent.verbose > 1
            ProgressMeter.next!(
                progress_meter;
                step = n_steps * n_envs,
                showvalues = [
                    ("explained_variance", explained_variance),
                    ("entropy_loss", total_entropy_losses[i]),
                    ("policy_loss", total_policy_losses[i]),
                    ("value_loss", total_value_losses[i]),
                    ("approx_kl_div", total_approx_kl_divs[i]),
                    ("clip_fraction", total_clip_fractions[i]),
                    ("loss", total_losses[i]),
                    ("fps", total_fps[i]),
                    ("grad_norm", total_grad_norms[i]),
                    ("learning_rate", learning_rate),
                ]
            )
        elseif agent.verbose > 0
            ProgressMeter.next!(progress_meter, step = n_steps * n_envs)
        end

        log_scalar!(agent.logger, "train/entropy_loss", total_entropy_losses[i])
        log_scalar!(agent.logger, "train/explained_variance", explained_variance)
        log_scalar!(agent.logger, "train/policy_loss", total_policy_losses[i])
        log_scalar!(agent.logger, "train/value_loss", total_value_losses[i])
        log_scalar!(agent.logger, "train/approx_kl_div", total_approx_kl_divs[i])
        log_scalar!(agent.logger, "train/clip_fraction", total_clip_fractions[i])
        log_scalar!(agent.logger, "train/loss", total_losses[i])
        log_scalar!(agent.logger, "train/grad_norm", total_grad_norms[i])
        log_scalar!(agent.logger, "train/learning_rate", learning_rate)
        if haskey(train_state.parameters, :log_std)
            log_scalar!(agent.logger, "train/std", mean(exp.(train_state.parameters[:log_std])))
        end
    end
    agent.train_state = train_state

    learn_stats = Dict(
        "entropy_losses" => total_entropy_losses,
        "policy_losses" => total_policy_losses,
        "value_losses" => total_value_losses,
        "approx_kl_divs" => total_approx_kl_divs,
        "clip_fractions" => total_clip_fractions,
        "losses" => total_losses,
        "explained_variances" => total_explained_variances,
        "fps" => total_fps,
        "grad_norms" => total_grad_norms,
        "learning_rates" => learning_rates
    )
    if !isnothing(callbacks)
        @timeit to "callback: training_end" begin
            if !all(c -> on_training_end(c, Base.@locals), callbacks)
                @warn "Training stopped due to callback failure"
                return nothing
            end
        end
    end
    if agent.verbose ≥ 2
        print_timer(to)
    end
    return learn_stats, to
end

function normalize(advantages::Vector{T}) where {T}
    mean_adv = mean(advantages)
    std_adv = std(advantages)
    epsilon = T(1.0e-8)
    norm_advantages = (advantages .- mean_adv) ./ (std_adv + epsilon)
    return norm_advantages
end

function clip_range!(values::Vector{T}, old_values::Vector{T}, clip_range::T) where {T}
    for i in eachindex(values)
        diff = values[i] - old_values[i]
        clipped_diff = clamp(diff, -clip_range, clip_range)
        values[i] = old_values[i] + clipped_diff
    end
    return nothing
end

function clip_range(old_values::Vector{T}, values::Vector{T}, clip_range::T) where {T}
    return old_values .+ clamp(values .- old_values, -clip_range, clip_range)
end


#TODO: vectorize this?
function normalize!(values::Vector{T}) where {T}
    mean_values = mean(values)
    std_values = std(values)
    epsilon = T(1.0e-8)
    values .= (values .- mean_values) ./ (std_values + epsilon)
    return nothing
end

function maybe_normalize!(advantages::Vector{T}, alg::PPO{T}) where {T}
    if alg.normalize_advantage
        normalize!(advantages)
    end
    return nothing
end

function (alg::PPO{T})(policy::AbstractActorCriticLayer, ps, st, batch_data) where {T}
    observations = batch_data[1]
    actions = batch_data[2]
    advantages::Vector{T} = batch_data[3]
    returns = batch_data[4]
    old_logprobs = batch_data[5]
    old_values = batch_data[6]

    # advantages = @ignore_derivatives alg.normalize_advantage ? normalize(advantages) : advantages
    #TODO: do we need to ignore derivatives here?
    @ignore_derivatives maybe_normalize!(advantages, alg)

    values, log_probs, entropy, st = evaluate_actions(policy, observations, actions, ps, st)
    values = !isnothing(alg.clip_range_vf) ? clip_range(old_values, values, alg.clip_range_vf::T) : values

    r = exp.(log_probs - old_logprobs)
    ratio_clipped = clamp.(r, 1 - alg.clip_range, 1 + alg.clip_range)
    p_loss = -mean(min.(r .* advantages, ratio_clipped .* advantages))
    ent_loss = -mean(entropy)

    v_loss = mean((values .- returns) .^ 2)
    loss = p_loss + alg.ent_coef * ent_loss + alg.vf_coef * v_loss

    stats = @ignore_derivatives begin
        # Calculate statistics
        clip_fraction = mean(r .!= ratio_clipped)
        #approx kl div
        log_ratio = log_probs - old_logprobs
        approx_kl_div = mean(exp.(log_ratio) .- 1 .- log_ratio)

        Dict{String, Any}(
            "policy_loss" => p_loss,
            "value_loss" => v_loss,
            "entropy_loss" => ent_loss,
            "clip_fraction" => clip_fraction,
            "approx_kl_div" => approx_kl_div,
            "entropy" => mean(entropy),
            "ratio" => mean(r)
        )
    end

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

# Helper function to process actions: convert from 1-based indexing to action space range
function process_action(action::Integer, action_space::Discrete, ::PPO)
    # Make sure its in valid range
    @assert action_space.start ≤ action ≤ action_space.start + action_space.n - 1
    return action
end
