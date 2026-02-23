using Test
using Drill
using TestItems


@testitem "DiscreteActorCriticLayer construction" tags = [:policies, :discrete, :construction] setup = [SharedTestSetup] begin
    using Random
    using Lux

    # Test basic construction
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Discrete(4, 0)  # 0-based (like Gymnasium)

    policy = DiscreteActorCriticLayer(obs_space, action_space)

    @test policy isa DiscreteActorCriticLayer
    @test isequal(policy.observation_space, obs_space)
    @test isequal(policy.action_space, action_space)
    @test typeof(policy) <: DiscreteActorCriticLayer{<:Any, <:Any, SharedFeatures}

    # Test with custom parameters
    policy_custom = DiscreteActorCriticLayer(
        obs_space, action_space;
        hidden_dims = [32, 16],
        activation = relu,
        shared_features = false
    )
    @test typeof(policy_custom) <: DiscreteActorCriticLayer{<:Any, <:Any, SeparateFeatures}

    # Test with 1-based action space
    action_space_1 = Discrete(3, 1)  # 1-based
    policy_1based = DiscreteActorCriticLayer(obs_space, action_space_1)
    @test policy_1based.action_space == action_space_1
end

@testitem "DiscreteActorCriticLayer parameter initialization" tags = [:policies, :discrete, :parameters] setup = [SharedTestSetup] begin
    using Random
    using Lux
    using ComponentArrays

    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Discrete(5, 0)
    policy = DiscreteActorCriticLayer(obs_space, action_space)

    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, policy)

    # Check parameter structure
    @test haskey(params, :feature_extractor)
    @test haskey(params, :actor_head)
    @test haskey(params, :critic_head)
    @test !haskey(params, :log_std)  # Discrete policies shouldn't have log_std

    # Check parameter count
    param_count = Lux.parameterlength(policy)
    @test param_count > 0
    @test param_count == length(ComponentVector(params))

    # Test state initialization
    states = Lux.initialstates(rng, policy)
    @test haskey(states, :feature_extractor)
    @test haskey(states, :actor_head)
    @test haskey(states, :critic_head)
end

@testitem "DiscreteActorCriticLayer prediction" tags = [:policies, :discrete, :prediction] setup = [SharedTestSetup] begin
    using Random
    using Lux

    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Discrete(3, 0)  # Actions: 0, 1, 2
    policy = DiscreteActorCriticLayer(obs_space, action_space)

    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, policy)
    states = Lux.initialstates(rng, policy)

    # Test single observation prediction (with batch dimension)
    obs = Float32[0.5, -0.3]  # Single observation as column vector
    batched_obs = stack([obs])
    actions, new_states = Drill.predict_actions(policy, batched_obs, params, states; deterministic = false, rng = rng)

    # Actions should be in environment action space after processing
    @test actions[1] ∈ action_space
    @test actions[1] isa Integer

    # Test deterministic prediction
    actions_det, _ = Drill.predict_actions(policy, batched_obs, params, states; deterministic = true, rng = rng)
    @test actions_det[1] ∈ action_space
    @test actions_det[1] isa Integer

    # Test batch prediction
    batch_obs = Float32[0.5 -0.2; -0.3 0.7]  # 2 observations
    batch_actions, _ = Drill.predict_actions(policy, batch_obs, params, states; deterministic = false, rng = rng)

    @test length(batch_actions) == 2
    @test all(a -> a ∈ action_space, batch_actions)
    @test all(a -> a isa Integer, batch_actions)
end

@testitem "DiscreteActorCriticLayer action evaluation" tags = [:policies, :discrete, :evaluation] setup = [SharedTestSetup] begin
    using Random
    using Lux
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Discrete(4, 0)  # Actions: 0, 1, 2, 3
    policy = DiscreteActorCriticLayer(obs_space, action_space)

    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, policy)
    states = Lux.initialstates(rng, policy)

    # Test single observation evaluation
    obs = Float32[0.5, -0.3]

    # Get action from policy (this will be in 1-based Julia indexing internally)
    batched_obs = stack([obs])
    actions, values, log_probs, _ = policy(batched_obs, params, states)

    # Test that actions are valid indices (1-based for internal use)
    @test actions[1] isa Integer
    @test 0 <= actions[1] <= action_space.n - 1

    # Evaluate the same actions
    eval_values, eval_log_probs, entropy, _ = Drill.evaluate_actions(policy, batched_obs, actions, params, states)

    # Values should match
    @test eval_values ≈ values atol = 1.0e-6

    # Log probabilities should match (approximately due to floating point)
    @test isapprox.(eval_log_probs, log_probs, atol = 1.0e-5) |> all

    # Entropy should be positive for stochastic policy
    @test entropy[1] >= 0.0f0

    # Test batch evaluation
    batch_obs = Float32[0.5 -0.2; -0.3 0.7]
    batch_actions, batch_values, batch_log_probs, _ = policy(batch_obs, params, states)

    eval_batch_values, eval_batch_log_probs, batch_entropy, _ = Drill.evaluate_actions(policy, batch_obs, batch_actions, params, states)

    @test length(eval_batch_values) == 2
    @test length(eval_batch_log_probs) == 2
    @test length(batch_entropy) == 2
    @test eval_batch_values ≈ batch_values atol = 1.0e-6
    @test all(eval_batch_log_probs .≈ batch_log_probs)
end

@testitem "DiscreteActorCriticLayer indexing consistency" tags = [:policies, :discrete, :indexing] setup = [SharedTestSetup] begin
    using Random
    using Lux
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

    # Test different action space configurations
    spaces_to_test = [
        Discrete(3, 0),   # 0-based (Gymnasium style): 0, 1, 2
        Discrete(3, 1),   # 1-based (Julia style): 1, 2, 3
        Discrete(4, -1),  # Custom start: -1, 0, 1, 2
    ]

    for action_space in spaces_to_test
        policy = DiscreteActorCriticLayer(obs_space, action_space)

        rng = Random.MersenneTwister(42)
        params = Lux.initialparameters(rng, policy)
        states = Lux.initialstates(rng, policy)

        obs = Float32[0.5, -0.3]
        batched_obs = stack([obs])

        # Test that policy actions (before processing) are in 1-based indexing
        actions, _, _, _ = policy(batched_obs, params, states)
        @test actions[1] ∈ action_space

        # Test that predict_actions() returns processed actions in action space range
        processed_actions, _ = Drill.predict_actions(policy, batched_obs, params, states)
        @test processed_actions[1] ∈ action_space

        # Test that evaluation works with stored actions (1-based)
        eval_values, eval_log_probs, entropy, _ = Drill.evaluate_actions(policy, batched_obs, actions, params, states)
        @test length(eval_log_probs) == 1
        @test length(entropy) == 1
        @test eval_log_probs[1] isa Float32
        @test entropy[1] >= 0.0f0
    end
end


@testitem "DiscreteActorCriticLayer vs ContinuousActorCriticLayer interface" tags = [:policies, :discrete, :interface] setup = [SharedTestSetup] begin
    using Random
    using Lux
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    discrete_action_space = Discrete(4, 0)
    continuous_action_space = Box(Float32[-1.0], Float32[1.0])

    discrete_policy = DiscreteActorCriticLayer(obs_space, discrete_action_space)
    continuous_policy = ContinuousActorCriticLayer(obs_space, continuous_action_space)

    # Test that both policies implement the same interface
    rng = Random.MersenneTwister(42)

    discrete_params = Lux.initialparameters(rng, discrete_policy)
    discrete_states = Lux.initialstates(rng, discrete_policy)

    continuous_params = Lux.initialparameters(rng, continuous_policy)
    continuous_states = Lux.initialstates(rng, continuous_policy)

    obs = Float32[0.5, -0.3]
    batched_obs = stack([obs])

    # Test that both implement the same methods
    discrete_actions, discrete_values, discrete_log_probs, _ = discrete_policy(batched_obs, discrete_params, discrete_states)
    continuous_actions, continuous_values, continuous_log_probs, _ = continuous_policy(batched_obs, continuous_params, continuous_states)

    # Test predict
    discrete_pred, _ = Drill.predict_actions(discrete_policy, batched_obs, discrete_params, discrete_states)
    continuous_pred, _ = Drill.predict_actions(continuous_policy, batched_obs, continuous_params, continuous_states)

    # Test predict_values
    discrete_vals, _ = predict_values(discrete_policy, batched_obs, discrete_params, discrete_states)
    continuous_vals, _ = predict_values(continuous_policy, batched_obs, continuous_params, continuous_states)

    # Test evaluate_actions
    discrete_eval_values, discrete_eval_log_probs, discrete_entropy, _ = Drill.evaluate_actions(discrete_policy, batched_obs, discrete_actions, discrete_params, discrete_states)
    continuous_eval_values, continuous_eval_log_probs, continuous_entropy, _ = Drill.evaluate_actions(continuous_policy, batched_obs, stack(continuous_actions), continuous_params, continuous_states)

    # Test that outputs have expected types and shapes
    @test discrete_actions isa Vector{<:Integer}
    @test continuous_actions isa Vector{<:Vector{<:Real}}
    @test discrete_pred isa Vector{<:Integer}
    @test continuous_pred isa Vector{<:Vector{<:Real}}
    @test discrete_vals isa Vector{<:Real}
    @test continuous_vals isa Vector{<:Real}
    @test length(discrete_eval_log_probs) == 1
    @test length(continuous_eval_log_probs) == 1
end

@testitem "DiscreteActorCriticLayer edge cases" tags = [:policies, :discrete, :edge_cases] setup = [SharedTestSetup] begin
    using Random
    using Lux
    # Test single action space
    obs_space = Box(Float32[-1.0], Float32[1.0])
    single_action_space = Discrete(1, 0)  # Only action 0
    policy = DiscreteActorCriticLayer(obs_space, single_action_space)

    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, policy)
    states = Lux.initialstates(rng, policy)

    obs = Float32[0.5]
    batched_obs = stack([obs])
    # Test that single action space works
    actions, values, log_probs, _ = policy(batched_obs, params, states)
    @test actions[1] == 0

    processed_action, _ = Drill.predict_actions(policy, batched_obs, params, states)
    @test processed_action[1] == 0

    # Test large action space
    large_action_space = Discrete(100, 0)
    large_policy = DiscreteActorCriticLayer(obs_space, large_action_space)

    large_params = Lux.initialparameters(rng, large_policy)
    large_states = Lux.initialstates(rng, large_policy)

    large_actions, _, _, _ = large_policy(batched_obs, large_params, large_states)
    @test large_actions[1] ∈ large_action_space

    large_processed, _ = Drill.predict_actions(large_policy, batched_obs, large_params, large_states)
    @test large_processed[1] ∈ large_action_space

    # Test negative start action space
    neg_action_space = Discrete(5, -2)  # Actions: -2, -1, 0, 1, 2
    neg_policy = DiscreteActorCriticLayer(obs_space, neg_action_space)

    neg_params = Lux.initialparameters(rng, neg_policy)
    neg_states = Lux.initialstates(rng, neg_policy)

    neg_actions, _, _, _ = neg_policy(batched_obs, neg_params, neg_states)
    @test neg_actions[1] ∈ neg_action_space

    neg_processed, _ = Drill.predict_actions(neg_policy, batched_obs, neg_params, neg_states)
    @test neg_processed[1] ∈ neg_action_space
end

@testitem "Basic Q-value actor critic policy" tags = [:policies, :qvalue, :actor_critic] setup = [SharedTestSetup] begin
    #test that the different q networks are different and produce different values
    using Random
    using Lux
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Box(Float32[-1.0], Float32[1.0])
    policy = ContinuousActorCriticLayer(
        obs_space, action_space, activation = relu,
        critic_type = QCritic(), shared_features = false
    )

    rng = Random.MersenneTwister(42)
    ps, st = Lux.setup(rng, policy)

    mock_obs = rand(Float32, 2, 10)
    mock_actions = rand(Float32, 1, 10)
    mock_values, st = predict_values(policy, mock_obs, mock_actions, ps, st)
    @test size(mock_values) == (2, 10)
    @test all(mock_values[1, :] .!= mock_values[2, :]) #test that the two networks are different

    actions, log_probs, st = action_log_prob(policy, mock_obs, ps, st)
    @test size(actions) == (10,)

end
