"""
    SAC{T <: AbstractFloat, E <: AbstractEntropyCoefficient} <: OffPolicyAlgorithm

Soft Actor-Critic algorithm with automatic entropy tuning.

# Fields
- `learning_rate`: Optimizer learning rate (default: 3e-4)
- `buffer_capacity`: Replay buffer size (default: 1M)
- `start_steps`: Random exploration steps before training (default: 100)
- `batch_size`: Batch size for updates (default: 256)
- `tau`: Soft update coefficient for target networks (default: 0.005)
- `gamma`: Discount factor (default: 0.99)
- `train_freq`: Steps between gradient updates (default: 1)
- `gradient_steps`: Gradient steps per update, -1 for auto (default: 1)
- `ent_coef`: Entropy coefficient (`AutoEntropyCoefficient` or `FixedEntropyCoefficient`)

# Example
```julia
sac = SAC(learning_rate=3f-4, buffer_capacity=1_000_000)
model = SACLayer(obs_space, act_space)
agent = Agent(model, sac)
train!(agent, env, sac, 500_000)
```
"""
@kwdef struct SAC{T <: AbstractFloat, E <: AbstractEntropyCoefficient} <: OffPolicyAlgorithm
    learning_rate::T = 3.0f-4 #learning rate
    buffer_capacity::Int = 1_000_000
    start_steps::Int = 100 # how many total steps to collect with random actions before first gradient update
    batch_size::Int = 256
    tau::T = 0.005f0 #soft update rate
    gamma::T = 0.99f0 #discount
    train_freq::Int = 1 # how many steps to take between gradient updates (train_freq*n_envs data points)
    gradient_steps::Int = 1 # number of gradient updates per rollout, -1 to do as many updates as steps (train_freq*n_envs)
    ent_coef::E = AutoEntropyCoefficient()
    target_update_interval::Int = 1 # how often to update the target networks
end

# Traits and adapter selection
action_adapter(::SAC, ::Box) = TanhScaleAdapter()
has_twin_critics(::SAC) = true
has_target_networks(::SAC) = true
has_entropy_tuning(::SAC) = true
uses_replay(::SAC) = true
critic_type(::SAC) = QCritic()

# Helper function to calculate target entropy for automatic entropy coefficient
function get_target_entropy(ent_coef::AutoEntropyCoefficient{T}, action_space) where {T}
    if ent_coef.target isa AutoEntropyTarget
        # For continuous action spaces, target entropy is typically -dim(action_space)
        return -T(prod(size(action_space)))
    elseif ent_coef.target isa FixedEntropyTarget
        return ent_coef.target.target
    else
        error("Unknown entropy target type: $(typeof(ent_coef.target))")
    end
end

function get_gradient_steps(alg::SAC, train_freq::Int = alg.train_freq, n_envs::Int = 1)
    if alg.gradient_steps == -1
        return train_freq * n_envs
    else
        return alg.gradient_steps
    end
end

get_target_entropy(ent_coef::FixedEntropyCoefficient, action_space) = nothing

function SACLayer(
        observation_space::Union{Discrete, Box{T}},
        action_space::Box{T};
        log_std_init::T = T(-3),
        hidden_dims = [512, 512],
        activation = relu,
        shared_features::Bool = false,
        critic_type::CriticType = QCritic()
    ) where {T}
    return ContinuousActorCriticLayer(
        observation_space, action_space;
        log_std_init, hidden_dims, activation, shared_features, critic_type
    )
end

function sac_ent_coef_loss(
        ::SAC,
        layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, QCritic}, ps, st, data;
        rng::AbstractRNG = Random.default_rng()
    )
    log_ent_coef = first(ps.log_ent_coef)
    layer_ps = data.layer_ps
    layer_st = data.layer_st
    _, log_probs_pi, layer_st = action_log_prob(layer, data.observations, layer_ps, layer_st; rng)
    target_entropy = data.target_entropy
    loss = -(log_ent_coef * @ignore_derivatives(log_probs_pi .+ target_entropy |> mean))
    return loss, st, NamedTuple()
end

function sac_actor_loss(
        ::SAC, layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, QCritic},
        ps, st, data; rng::AbstractRNG = Random.default_rng()
    )
    obs = data.observations
    ent_coef = exp(first(data.log_ent_coef.log_ent_coef))
    actions_pi, log_probs_pi, st = action_log_prob(layer, obs, ps, st; rng)
    q_values, st = predict_values(layer, obs, actions_pi, ps, st)
    min_q_values = minimum(q_values, dims = 1) |> vec
    loss = mean(ent_coef .* log_probs_pi - min_q_values)
    return loss, st, NamedTuple()
end

function sac_critic_loss(
        alg::SAC, layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, QCritic}, ps, st, data;
        rng::AbstractRNG = Random.default_rng()
    )
    obs, actions, rewards, terminated, _, next_obs = data.observations, data.actions, data.rewards, data.terminated, data.truncated, data.next_observations
    gamma = alg.gamma
    ent_coef = exp(first(data.log_ent_coef.log_ent_coef))
    target_ps = data.target_ps
    target_st = data.target_st

    # Current Q-values
    current_q_values, new_st = predict_values(layer, obs, actions, ps, st)

    # Target Q-values (no gradients)
    obs_dims = ndims(obs)
    next_obs = selectdim(next_obs, obs_dims, .!terminated)
    @assert !any(isnan, next_obs) "Next observations contain NaNs"
    target_q_values = @ignore_derivatives begin
        next_actions, next_log_probs, st = action_log_prob(layer, next_obs, ps, st; rng)
        #replace critic ps and st with target
        ps_with_target = merge_params(ps, target_ps)
        st_with_target = merge(st, target_st)
        next_q_vals, _ = predict_values(layer, next_obs, next_actions, ps_with_target, st_with_target)
        min_next_q = minimum(next_q_vals, dims = 1) |> vec

        # Add entropy term
        next_q_vals_with_entropy = min_next_q .- ent_coef .* next_log_probs
        target_q_vals = copy(rewards)
        # Bellman target
        target_q_vals[.!terminated] .+= gamma .* next_q_vals_with_entropy
        target_q_vals
    end

    # Critic loss (sum over all Q-networks)
    T = eltype(current_q_values)
    #TODO: type stability here?
    critic_loss = T(0.5) * sum(mean((current_q .- target_q_values) .^ 2) for current_q in eachrow(current_q_values))

    stats = (mean_q_values = mean(current_q_values),)
    return critic_loss, new_st, stats
end

# Callable functions for SAC algorithm - needed for Lux.Training.compute_gradients
function (alg::SAC)(::ContinuousActorCriticLayer, ps, st, batch_data)
    # This is the combined loss function for all networks
    # In practice, we'll compute separate losses for actor, critic, and entropy coefficient
    error("SAC algorithm object should not be called directly. Use specific loss functions instead.")
end

function Agent(
        layer::ContinuousActorCriticLayer,
        alg::SAC;
        optimizer_type::Type{<:Optimisers.AbstractRule} = Optimisers.Adam,
        logger = NoTrainingLogger(),
        stats_window::Int = 100,
        rng::AbstractRNG = Random.default_rng(),
        verbose::Int = 1
    )
    ps, st = Lux.setup(rng, layer)
    optimizer = make_optimizer(optimizer_type, alg)
    train_state = Lux.Training.TrainState(layer, ps, st, optimizer)
    Q_target_parameters = copy_critic_parameters(layer, ps)
    Q_target_states = copy_critic_states(layer, st)

    # Always initialize entropy coefficient train state
    ent_coef_params = init_entropy_coefficient(alg.ent_coef)
    ent_optimizer = make_optimizer(optimizer_type, alg)
    ent_train_state = Lux.Training.TrainState(layer, ent_coef_params, NamedTuple(), ent_optimizer)

    adapter = action_adapter(alg, action_space(layer))
    aux = QAux(Q_target_parameters, Q_target_states, ent_train_state)

    logger = convert(AbstractTrainingLogger, logger)
    return Agent(
        layer, alg, adapter, train_state, optimizer_type, stats_window, logger, verbose, rng,
        AgentStats(0, 0), aux
    )
end


function copy_critic_parameters(layer::ContinuousActorCriticLayer{<:Any, <:Any, N, QCritic, SharedFeatures}, ps::NamedTuple) where {N <: AbstractNoise}
    return (feature_extractor = deepcopy(ps.feature_extractor), critic_head = deepcopy(ps.critic_head))
end

function copy_critic_parameters(layer::ContinuousActorCriticLayer{<:Any, <:Any, N, QCritic, SeparateFeatures}, ps::NamedTuple) where {N <: AbstractNoise}
    return (critic_feature_extractor = deepcopy(ps.critic_feature_extractor), critic_head = deepcopy(ps.critic_head))
end

function copy_critic_states(layer::ContinuousActorCriticLayer{<:Any, <:Any, N, QCritic, SharedFeatures}, st::NamedTuple) where {N <: AbstractNoise}
    return (feature_extractor = deepcopy(st.feature_extractor), critic_head = deepcopy(st.critic_head))
end

function copy_critic_states(layer::ContinuousActorCriticLayer{<:Any, <:Any, N, QCritic, SeparateFeatures}, st::NamedTuple) where {N <: AbstractNoise}
    return (critic_feature_extractor = deepcopy(st.critic_feature_extractor), critic_head = deepcopy(st.critic_head))
end

function init_entropy_coefficient(entropy_coefficient::FixedEntropyCoefficient)
    return (; log_ent_coef = [entropy_coefficient.coef |> log])
end

function init_entropy_coefficient(entropy_coefficient::AutoEntropyCoefficient)
    return (; log_ent_coef = [entropy_coefficient.initial_value |> log])
end

function predict_actions(
        agent::Agent{<:ContinuousActorCriticLayer, <:SAC, <:AbstractActionAdapter, <:AbstractRNG, <:AbstractTrainingLogger, <:Any},
        observations::AbstractVector;
        deterministic::Bool = false,
        rng::AbstractRNG = agent.rng,
        raw::Bool = false
    )
    #TODO add !to name?
    train_state = agent.train_state
    layer = agent.layer
    ps = train_state.parameters
    st = train_state.states
    batched_obs = batch(observations, observation_space(layer))
    actions, st = predict_actions(layer, batched_obs, ps, st; deterministic, rng)
    @reset train_state.states = st
    agent.train_state = train_state
    if raw
        return actions
    else
        adapter = agent.action_adapter
        return to_env.(Ref(adapter), actions, Ref(action_space(layer)))
    end
    return actions
end

# collect_trajectories and collect_rollout! are now in buffers/off_policy_collection.jl

# Unified Agent supports raw actions for off-policy collection when using SAC
function predict_actions_raw(
        agent::Agent{<:ContinuousActorCriticLayer, <:SAC, <:AbstractActionAdapter, <:AbstractRNG, <:AbstractTrainingLogger, <:Any},
        observations::AbstractVector
    )
    return predict_actions(agent, observations; raw = true)
end

# Clean logging structure for SAC
struct SACTrainingStats{T <: AbstractFloat}
    actor_losses::Vector{T}
    critic_losses::Vector{T}
    entropy_losses::Vector{T}
    entropy_coefficients::Vector{T}
    q_values::Vector{T}
    learning_rates::Vector{T}
    grad_norms::Vector{T}
    fps::Vector{T}
    steps_taken::Vector{Int}
end

function SACTrainingStats{T}() where {T <: AbstractFloat}
    return SACTrainingStats{T}(T[], T[], T[], T[], T[], T[], T[], T[], T[])
end

function log_sac_training!(
        agent::Agent{<:ContinuousActorCriticLayer, <:SAC, <:AbstractActionAdapter, <:AbstractRNG, <:AbstractTrainingLogger, <:Any},
        stats::SACTrainingStats,
        step::Int,
        env::AbstractParallelEnv
    )
    set_step!(agent.logger, step)
    if !isempty(stats.actor_losses)
        log_scalar!(agent.logger, "train/actor_loss", stats.actor_losses[end])
    end
    if !isempty(stats.critic_losses)
        log_scalar!(agent.logger, "train/critic_loss", stats.critic_losses[end])
    end
    if !isempty(stats.entropy_losses)
        log_scalar!(agent.logger, "train/entropy_loss", stats.entropy_losses[end])
    end
    if !isempty(stats.entropy_coefficients)
        log_scalar!(agent.logger, "train/entropy_coefficient", stats.entropy_coefficients[end])
    end
    if !isempty(stats.q_values)
        log_scalar!(agent.logger, "train/q_values", stats.q_values[end])
    end
    if !isempty(stats.learning_rates)
        log_scalar!(agent.logger, "train/learning_rate", stats.learning_rates[end])
    end
    if !isempty(stats.grad_norms)
        log_scalar!(agent.logger, "train/grad_norm", stats.grad_norms[end])
    end
    if !isempty(stats.fps)
        log_scalar!(agent.logger, "env/fps", stats.fps[end])
    end
    log_scalar!(agent.logger, "train/total_steps", steps_taken(agent))

    # Log mean std (exp of log_std) if present, consistent with PPO logging
    if haskey(agent.train_state.parameters, :log_std)
        mean_std = Statistics.mean(exp.(agent.train_state.parameters[:log_std]))
        log_scalar!(agent.logger, "train/std", mean_std)
    end

    # Log episode statistics
    log_stats(env, agent.logger)
    return nothing
end

"""
    update!(agent, alg::SAC, batch_data; ad_type)

Perform a single SAC gradient update step (entropy, critic, actor, targets) and update agent state.
Returns a NamedTuple of useful scalars from this step.
"""
function update!(
        agent::Agent{<:ContinuousActorCriticLayer, <:SAC, <:AbstractActionAdapter, <:AbstractRNG, <:AbstractTrainingLogger, <:Any},
        alg::SAC,
        batch_data;
        ad_type::Lux.Training.AbstractADType = AutoZygote()
    )
    layer = agent.layer
    train_state = agent.train_state
    # Entropy coefficient update if enabled
    ent_loss = nothing
    if alg.ent_coef isa AutoEntropyCoefficient
        target_entropy = get_target_entropy(alg.ent_coef, action_space(layer))
        ent_train_state = agent.aux.ent_train_state
        ent_data = (
            observations = batch_data.observations,
            layer_ps = train_state.parameters,
            layer_st = train_state.states,
            target_entropy = target_entropy,
            target_ps = agent.aux.Q_target_parameters,
            target_st = agent.aux.Q_target_states,
        )
        ent_grad, ent_loss_val, _, ent_train_state = Lux.Training.compute_gradients(
            ad_type,
            (model, ps, st, data) -> sac_ent_coef_loss(alg, layer, ps, st, data; rng = agent.rng),
            ent_data,
            ent_train_state
        )
        ent_train_state = Lux.Training.apply_gradients!(ent_train_state, ent_grad)
        agent.aux.ent_train_state = ent_train_state
        ent_loss = ent_loss_val
    end

    # Critic update
    critic_data = (
        observations = batch_data.observations,
        actions = batch_data.actions,
        rewards = batch_data.rewards,
        terminated = batch_data.terminated,
        truncated = batch_data.truncated,
        next_observations = batch_data.next_observations,
        log_ent_coef = agent.aux.ent_train_state.parameters,
        target_ps = agent.aux.Q_target_parameters,
        target_st = agent.aux.Q_target_states,
    )
    critic_grad, critic_loss, critic_stats, train_state = Lux.Training.compute_gradients(
        ad_type,
        (model, ps, st, data) -> sac_critic_loss(alg, layer, ps, st, data; rng = agent.rng),
        critic_data,
        train_state
    )
    train_state = Lux.Training.apply_gradients(train_state, critic_grad)

    # Actor update
    actor_data = (
        observations = batch_data.observations,
        actions = batch_data.actions,
        rewards = batch_data.rewards,
        terminated = batch_data.terminated,
        truncated = batch_data.truncated,
        next_observations = batch_data.next_observations,
        log_ent_coef = agent.aux.ent_train_state.parameters,
    )
    actor_loss_grad, actor_loss, _, train_state = Lux.Training.compute_gradients(
        ad_type,
        (model, ps, st, data) -> sac_actor_loss(alg, layer, ps, st, data; rng = agent.rng),
        actor_data,
        train_state
    )
    zero_critic_grads!(actor_loss_grad, layer)
    train_state = Lux.Training.apply_gradients(train_state, actor_loss_grad)

    # Target networks update
    if agent.stats.gradient_updates % alg.target_update_interval == 0
        agent.aux.Q_target_states = copy_critic_states(layer, train_state.states)
        polyak_update!(agent.aux.Q_target_parameters, train_state.parameters, alg.tau)
    end

    agent.train_state = train_state

    current_ent_coef = exp(first(agent.aux.ent_train_state.parameters.log_ent_coef))
    T = eltype(alg.learning_rate)
    total_grad_norm = sqrt(nested_norm(critic_grad, T)^2 + nested_norm(actor_loss_grad, T)^2)
    add_gradient_update!(agent)
    return (
        actor_loss = actor_loss,
        critic_loss = critic_loss,
        entropy_loss = ent_loss,
        mean_q_values = critic_stats.mean_q_values,
        entropy_coefficient = current_ent_coef,
        grad_norm = total_grad_norm,
    )
end

function train!(
        agent::Agent{<:ContinuousActorCriticLayer, <:SAC, <:AbstractActionAdapter, <:AbstractRNG, <:AbstractTrainingLogger, <:Any},
        env::AbstractParallelEnv,
        alg::SAC,
        max_steps::Int;
        kwargs...
    )
    replay_buffer = ReplayBuffer(observation_space(env), action_space(env), alg.buffer_capacity)
    return train!(agent, replay_buffer, env, alg, max_steps; kwargs...)
end

function train!(
        agent::Agent{<:ContinuousActorCriticLayer, <:SAC, <:AbstractActionAdapter, <:AbstractRNG, <:AbstractTrainingLogger, <:Any},
        replay_buffer::ReplayBuffer,
        env::AbstractParallelEnv,
        alg::SAC,
        max_steps::Int;
        #TODO: remove union?
        callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing,
        ad_type::Lux.Training.AbstractADType = AutoZygote()
    )
    to = TimerOutput()
    n_envs = number_of_envs(env)
    layer = agent.layer
    train_state = agent.train_state

    # Initialize training statistics
    T = eltype(alg.learning_rate)
    training_stats = SACTrainingStats{T}()

    # Calculate target entropy for automatic entropy coefficient
    target_entropy = get_target_entropy(alg.ent_coef, action_space(layer))

    # Check if we should update entropy coefficient
    update_entropy_coef = alg.ent_coef isa AutoEntropyCoefficient

    gradient_updates_performed = 0

    total_start_steps = alg.start_steps > 0 ? alg.start_steps : alg.train_freq * n_envs
    adjusted_total_start_steps = max(1, div(total_start_steps, n_envs)) * n_envs
    n_steps = div(adjusted_total_start_steps, n_envs)


    # Main training loop
    training_iteration = 0
    iterations = div(max_steps - adjusted_total_start_steps, alg.train_freq * n_envs) + 1

    total_steps = n_steps * n_envs + alg.train_freq * n_envs * (iterations - 1)

    progress_meter = Progress(
        total_steps, desc = "Training...",
        showspeed = true, enabled = agent.verbose > 0
    )

    agent.verbose > 0 && @info "Starting SAC training with buffer size: $(length(replay_buffer)),
    start_steps: $(alg.start_steps), train_freq: $(alg.train_freq), number_of_envs: $(n_envs),
    adjusted (total) start_steps: $(adjusted_total_start_steps), iterations: $(iterations), total_steps: $(total_steps)"

    # Callbacks: training start
    if !isnothing(callbacks)
        @timeit to "callback: training_start" begin
            if !all(c -> on_training_start(c, Base.@locals), callbacks)
                @warn "Training stopped due to callback failure"
                return agent, replay_buffer, training_stats
            end
        end
    end
    @timeit to "training_loop" for training_iteration in 1:iterations  # Adjust this termination condition as needed
        # @info "Training iteration $training_iteration, collecting rollout ($n_steps steps)"

        # Callbacks: rollout start
        if !isnothing(callbacks)
            @timeit to "callback: rollout_start" begin
                if !all(c -> on_rollout_start(c, Base.@locals), callbacks)
                    @warn "Training stopped due to callback failure"
                    return agent, replay_buffer, training_stats
                end
            end
        end

        # Collect experience
        fps, success = @timeit to "collect_rollout" collect_rollout!(
            replay_buffer, agent, alg, env, n_steps, progress_meter; callbacks,
            use_random_actions = training_iteration == 1 && alg.start_steps > 0
        )
        if !success
            @warn "Collecting rollout stopped due to callback failure"
            return agent, replay_buffer, training_stats
        end

        # Callbacks: rollout end
        if !isnothing(callbacks)
            @timeit to "callback: rollout_end" begin
                if !all(c -> on_rollout_end(c, Base.@locals), callbacks)
                    @warn "Training stopped due to callback failure"
                    return agent, replay_buffer, training_stats
                end
            end
        end
        #set steps to train_freq after first (potentially larger) rollout
        push!(training_stats.fps, fps)
        add_step!(agent, n_steps * n_envs)
        increment_step!(agent.logger, n_steps * n_envs)
        n_steps = alg.train_freq

        # Perform gradient updates
        n_updates = get_gradient_steps(alg, alg.train_freq, n_envs)
        data_loader = get_data_loader(replay_buffer, alg.batch_size, n_updates, true, true, agent.rng)

        @timeit to "gradient_updates" for (i, batch_data) in enumerate(data_loader)
            stats_step = update!(agent, alg, batch_data; ad_type)
            if stats_step.entropy_loss !== nothing
                push!(training_stats.entropy_losses, stats_step.entropy_loss)
            end
            push!(training_stats.critic_losses, stats_step.critic_loss)
            push!(training_stats.actor_losses, stats_step.actor_loss)
            push!(training_stats.q_values, stats_step.mean_q_values)
            push!(training_stats.entropy_coefficients, stats_step.entropy_coefficient)
            push!(training_stats.learning_rates, alg.learning_rate)
            push!(training_stats.grad_norms, stats_step.grad_norm)
            gradient_updates_performed += 1
        end

        # Log training statistics
        log_sac_training!(agent, training_stats, steps_taken(agent), env)
    end

    # Callbacks: training end
    if !isnothing(callbacks)
        @timeit to "callback: training_end" begin
            if !all(c -> on_training_end(c, Base.@locals), callbacks)
                @warn "Training stopped due to callback failure"
                return agent, replay_buffer, training_stats, to
            end
        end
    end

    # Return training statistics
    if agent.verbose ≥ 2
        print_timer(to)
    end
    return agent, replay_buffer, training_stats, to
end

function process_action(action, action_space::Box{T}, ::SAC) where {T}
    # First check if type conversion is needed
    if eltype(action) != T
        @warn "Action type mismatch: $(eltype(action)) != $T"
        action = convert.(T, action)
    end
    action = scale_to_space(action, action_space)
    return action
end
