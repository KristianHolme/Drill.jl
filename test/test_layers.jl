using Test
using Drill
using Random
using Lux
using ComponentArrays
include("setup.jl")
using .TestSetup

@testset "DiscreteActorCriticLayer construction" begin
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Discrete(4, 0)

    layer = DiscreteActorCriticLayer(obs_space, action_space)

    @test layer isa DiscreteActorCriticLayer
    @test isequal(layer.observation_space, obs_space)
    @test isequal(layer.action_space, action_space)
    @test typeof(layer) <: DiscreteActorCriticLayer{<:Any, <:Any, SharedFeatures}

    layer_custom = DiscreteActorCriticLayer(
        obs_space, action_space;
        hidden_dims = [32, 16],
        activation = relu,
        shared_features = false
    )
    @test typeof(layer_custom) <: DiscreteActorCriticLayer{<:Any, <:Any, SeparateFeatures}

    action_space_1 = Discrete(3, 1)
    layer_1based = DiscreteActorCriticLayer(obs_space, action_space_1)
    @test layer_1based.action_space == action_space_1
end

@testset "DiscreteActorCriticLayer parameter initialization" begin
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Discrete(5, 0)
    layer = DiscreteActorCriticLayer(obs_space, action_space)

    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, layer)

    @test haskey(params, :feature_extractor)
    @test haskey(params, :actor_head)
    @test haskey(params, :critic_head)
    @test !haskey(params, :log_std)

    param_count = Lux.parameterlength(layer)
    @test param_count > 0
    @test param_count == length(ComponentVector(params))

    states = Lux.initialstates(rng, layer)
    @test haskey(states, :feature_extractor)
    @test haskey(states, :actor_head)
    @test haskey(states, :critic_head)
end

@testset "DiscreteActorCriticLayer prediction" begin
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Discrete(3, 0)
    layer = DiscreteActorCriticLayer(obs_space, action_space)

    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, layer)
    states = Lux.initialstates(rng, layer)

    obs = Float32[0.5, -0.3]
    batched_obs = stack([obs])
    actions_onehot, new_states = Drill.predict_actions(layer, batched_obs, params, states; deterministic = false, rng = rng)

    actions = Drill.onehotbatch_to_discrete(actions_onehot, action_space)
    @test actions[1] ∈ action_space
    @test actions[1] isa Integer

    actions_det_onehot, _ = Drill.predict_actions(layer, batched_obs, params, states; deterministic = true, rng = rng)
    actions_det = Drill.onehotbatch_to_discrete(actions_det_onehot, action_space)
    @test actions_det[1] ∈ action_space
    @test actions_det[1] isa Integer

    batch_obs = Float32[0.5 -0.2; -0.3 0.7]
    batch_actions_onehot, _ = Drill.predict_actions(layer, batch_obs, params, states; deterministic = false, rng = rng)
    batch_actions = Drill.onehotbatch_to_discrete(batch_actions_onehot, action_space)

    @test length(batch_actions) == 2
    @test all(a -> a ∈ action_space, batch_actions)
    @test all(a -> a isa Integer, batch_actions)
end

@testset "DiscreteActorCriticLayer action evaluation" begin
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Discrete(4, 0)
    layer = DiscreteActorCriticLayer(obs_space, action_space)

    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, layer)
    states = Lux.initialstates(rng, layer)

    obs = Float32[0.5, -0.3]

    batched_obs = stack([obs])
    actions_onehot, values, log_probs, _ = layer(batched_obs, params, states)

    actions = Drill.onehotbatch_to_discrete(actions_onehot, action_space)
    @test actions[1] isa Integer
    @test actions[1] ∈ action_space

    eval_values, eval_log_probs, entropy, _ = Drill.evaluate_actions(layer, batched_obs, actions_onehot, params, states)

    @test eval_values ≈ values atol = 1.0e-6

    @test isapprox.(eval_log_probs, log_probs, atol = 1.0e-5) |> all

    @test entropy[1] >= 0.0f0

    batch_obs = Float32[0.5 -0.2; -0.3 0.7]
    batch_actions_onehot, batch_values, batch_log_probs, _ = layer(batch_obs, params, states)

    eval_batch_values, eval_batch_log_probs, batch_entropy, _ = Drill.evaluate_actions(layer, batch_obs, batch_actions_onehot, params, states)

    @test length(eval_batch_values) == 2
    @test length(eval_batch_log_probs) == 2
    @test length(batch_entropy) == 2
    @test eval_batch_values ≈ batch_values atol = 1.0e-6
    @test all(eval_batch_log_probs .≈ batch_log_probs)
end

@testset "DiscreteActorCriticLayer indexing consistency" begin
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

    spaces_to_test = [
        Discrete(3, 0),
        Discrete(3, 1),
        Discrete(4, -1),
    ]

    for action_space in spaces_to_test
        layer = DiscreteActorCriticLayer(obs_space, action_space)

        rng = Random.MersenneTwister(42)
        params = Lux.initialparameters(rng, layer)
        states = Lux.initialstates(rng, layer)

        obs = Float32[0.5, -0.3]
        batched_obs = stack([obs])

        actions_onehot, _, _, _ = layer(batched_obs, params, states)
        actions = Drill.onehotbatch_to_discrete(actions_onehot, action_space)
        @test actions[1] ∈ action_space

        processed_actions_onehot, _ = Drill.predict_actions(layer, batched_obs, params, states)
        processed_actions = Drill.onehotbatch_to_discrete(processed_actions_onehot, action_space)
        @test processed_actions[1] ∈ action_space

        eval_values, eval_log_probs, entropy, _ = Drill.evaluate_actions(layer, batched_obs, actions_onehot, params, states)
        @test length(eval_log_probs) == 1
        @test length(entropy) == 1
        @test eval_log_probs[1] isa Float32
        @test entropy[1] >= 0.0f0
    end
end

@testset "DiscreteActorCriticLayer vs ContinuousActorCriticLayer interface" begin
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    discrete_action_space = Discrete(4, 0)
    continuous_action_space = Box(Float32[-1.0], Float32[1.0])

    discrete_layer = DiscreteActorCriticLayer(obs_space, discrete_action_space)
    continuous_layer = ContinuousActorCriticLayer(obs_space, continuous_action_space)

    rng = Random.MersenneTwister(42)

    discrete_params = Lux.initialparameters(rng, discrete_layer)
    discrete_states = Lux.initialstates(rng, discrete_layer)

    continuous_params = Lux.initialparameters(rng, continuous_layer)
    continuous_states = Lux.initialstates(rng, continuous_layer)

    obs = Float32[0.5, -0.3]
    batched_obs = stack([obs])

    discrete_actions_onehot, discrete_values, discrete_log_probs, _ = discrete_layer(batched_obs, discrete_params, discrete_states)
    continuous_actions, continuous_values, continuous_log_probs, _ = continuous_layer(batched_obs, continuous_params, continuous_states)

    discrete_pred_onehot, _ = Drill.predict_actions(discrete_layer, batched_obs, discrete_params, discrete_states)
    continuous_pred, _ = Drill.predict_actions(continuous_layer, batched_obs, continuous_params, continuous_states)

    discrete_actions = Drill.onehotbatch_to_discrete(discrete_actions_onehot, discrete_action_space)
    discrete_pred = Drill.onehotbatch_to_discrete(discrete_pred_onehot, discrete_action_space)

    discrete_vals, _ = predict_values(discrete_layer, batched_obs, discrete_params, discrete_states)
    continuous_vals, _ = predict_values(continuous_layer, batched_obs, continuous_params, continuous_states)

    discrete_eval_values, discrete_eval_log_probs, discrete_entropy, _ = Drill.evaluate_actions(discrete_layer, batched_obs, discrete_actions_onehot, discrete_params, discrete_states)
    continuous_eval_values, continuous_eval_log_probs, continuous_entropy, _ = Drill.evaluate_actions(continuous_layer, batched_obs, continuous_actions, continuous_params, continuous_states)

    @test discrete_actions isa Vector{<:Integer}
    @test continuous_actions isa AbstractArray{<:Real}
    @test discrete_pred isa Vector{<:Integer}
    @test continuous_pred isa AbstractArray{<:Real}
    @test discrete_vals isa Vector{<:Real}
    @test continuous_vals isa Vector{<:Real}
    @test length(discrete_eval_log_probs) == 1
    @test length(continuous_eval_log_probs) == 1
end

@testset "DiscreteActorCriticLayer edge cases" begin
    obs_space = Box(Float32[-1.0], Float32[1.0])
    single_action_space = Discrete(1, 0)
    layer = DiscreteActorCriticLayer(obs_space, single_action_space)

    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, layer)
    states = Lux.initialstates(rng, layer)

    obs = Float32[0.5]
    batched_obs = stack([obs])
    actions_onehot, values, log_probs, _ = layer(batched_obs, params, states)
    actions = Drill.onehotbatch_to_discrete(actions_onehot, single_action_space)
    @test actions[1] == 0

    processed_action_onehot, _ = Drill.predict_actions(layer, batched_obs, params, states)
    processed_action = Drill.onehotbatch_to_discrete(processed_action_onehot, single_action_space)
    @test processed_action[1] == 0

    large_action_space = Discrete(100, 0)
    large_layer = DiscreteActorCriticLayer(obs_space, large_action_space)

    large_params = Lux.initialparameters(rng, large_layer)
    large_states = Lux.initialstates(rng, large_layer)

    large_actions_onehot, _, _, _ = large_layer(batched_obs, large_params, large_states)
    large_actions = Drill.onehotbatch_to_discrete(large_actions_onehot, large_action_space)
    @test large_actions[1] ∈ large_action_space

    large_processed_onehot, _ = Drill.predict_actions(large_layer, batched_obs, large_params, large_states)
    large_processed = Drill.onehotbatch_to_discrete(large_processed_onehot, large_action_space)
    @test large_processed[1] ∈ large_action_space

    neg_action_space = Discrete(5, -2)
    neg_layer = DiscreteActorCriticLayer(obs_space, neg_action_space)

    neg_params = Lux.initialparameters(rng, neg_layer)
    neg_states = Lux.initialstates(rng, neg_layer)

    neg_actions_onehot, _, _, _ = neg_layer(batched_obs, neg_params, neg_states)
    neg_actions = Drill.onehotbatch_to_discrete(neg_actions_onehot, neg_action_space)
    @test neg_actions[1] ∈ neg_action_space

    neg_processed_onehot, _ = Drill.predict_actions(neg_layer, batched_obs, neg_params, neg_states)
    neg_processed = Drill.onehotbatch_to_discrete(neg_processed_onehot, neg_action_space)
    @test neg_processed[1] ∈ neg_action_space
end

@testset "Basic Q-value actor critic layer" begin
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Box(Float32[-1.0], Float32[1.0])
    layer = ContinuousActorCriticLayer(
        obs_space, action_space, activation = relu,
        critic_type = QCritic(), shared_features = false
    )

    rng = Random.MersenneTwister(42)
    ps, st = Lux.setup(rng, layer)

    mock_obs = rand(Float32, 2, 10)
    mock_actions = rand(Float32, 1, 10)
    mock_values, st = predict_values(layer, mock_obs, mock_actions, ps, st)
    @test size(mock_values) == (2, 10)
    @test all(mock_values[1, :] .!= mock_values[2, :])

    actions, log_probs, st = action_log_prob(layer, mock_obs, ps, st)
    @test size(actions) == (1, 10)
end
