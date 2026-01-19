@testitem "SAC get_actions_from_features dimension mismatch detection" tags = [:sac, :dimensions] setup = [SharedTestSetup] begin
    using Random
    using Lux

    # Test the get_actions_from_features function directly with various feature dimensions
    # to catch reshape errors in the actor head network
    @testset "get_actions_from_features dimension mismatch detection" begin
        rng = Random.Xoshiro(42)

        # Test 1: Basic dimension compatibility
        @testset "Compatible feature dimensions" begin
            obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])  # 2D obs
            action_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])  # 2D action

            policy = ContinuousActorCriticLayer(obs_space, action_space; hidden_dims = [4, 4], critic_type = QCritic())
            ps = Lux.initialparameters(rng, policy)
            st = Lux.initialstates(rng, policy)

            # Test with different batch sizes
            for batch_size in [1, 2, 4, 8]
                obs = rand(obs_space, batch_size)
                batch_obs = DRiL.batch(obs, obs_space)
                actor_feats, critic_feats, st = DRiL.extract_features(policy, batch_obs, ps, st)
                @test_nowarn DRiL.get_actions_from_features(policy, actor_feats, ps, st)
            end
        end

        # Test 2: Feature dimension mismatches that could cause reshape errors
        @testset "Feature dimension mismatches" begin
            # Test case that replicates the original error pattern:
            # "DimensionMismatch: new dimensions (2, 4) must be consistent with array size 4"
            # This suggests trying to reshape 4 elements into (2, 4) = 8 elements

            @testset "Original error scenario: 2D obs -> 2D action" begin
                obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])  # 2D obs
                action_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])  # 2D action

                policy = ContinuousActorCriticLayer(obs_space, action_space; hidden_dims = [4, 4], critic_type = QCritic())
                ps = Lux.initialparameters(rng, policy)
                st = Lux.initialstates(rng, policy)

                # Test with feature matrices that could trigger reshape errors
                batch_size = 2
                feats = randn(Float32, 2, batch_size)  # Correct: 2D features for 2D observation space
                @test_nowarn DRiL.get_actions_from_features(policy, feats, ps, st)
            end

            @testset "Mismatched feature dimensions" begin
                obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])  # 2D obs
                action_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])  # 2D action

                policy = ContinuousActorCriticLayer(obs_space, action_space; hidden_dims = [6, 4], critic_type = QCritic())
                ps = Lux.initialparameters(rng, policy)
                st = Lux.initialstates(rng, policy)

                batch_size = 2

                # Wrong feature dimensions (should be 2D, not 4D)
                wrong_feats = randn(Float32, 4, batch_size)
                @test_throws Exception DRiL.get_actions_from_features(policy, wrong_feats, ps, st)

                # Wrong feature dimensions (should be 2D, not 1D)
                wrong_feats_1d = randn(Float32, 1, batch_size)
                @test_throws Exception DRiL.get_actions_from_features(policy, wrong_feats_1d, ps, st)
            end
        end

        # Test 3: Different observation/action dimension combinations
        @testset "Various obs/action dimension combinations" begin
            test_cases = [
                (1, 1), (1, 2), (1, 4),
                (2, 1), (2, 2), (2, 4),
                (4, 1), (4, 2), (4, 4),
                (8, 2), (8, 4),  # Including the original error case
            ]

            for (obs_dim, action_dim) in test_cases
                @testset "$(obs_dim)D obs, $(action_dim)D action" begin
                    obs_space = Box(Float32[-1.0], Float32[1.0], (obs_dim,))
                    action_space = Box(Float32[-1.0], Float32[1.0], (action_dim,))

                    # Adjust hidden dimensions based on input/output sizes
                    hidden_dim = max(obs_dim, action_dim) * 2
                    policy = ContinuousActorCriticLayer(
                        obs_space, action_space;
                        hidden_dims = [hidden_dim, hidden_dim],
                        critic_type = QCritic()
                    )
                    ps = Lux.initialparameters(rng, policy)
                    st = Lux.initialstates(rng, policy)

                    # Test with correct feature dimensions
                    batch_size = 2
                    feats = randn(Float32, obs_dim, batch_size)
                    actions, new_st = @test_nowarn DRiL.get_actions_from_features(policy, feats, ps, st)

                    # Verify output dimensions are correct
                    @test size(actions) == (action_dim, batch_size)
                end
            end
        end

        # Test 4: Batch size edge cases that could trigger reshape errors
        @testset "Batch size edge cases" begin
            obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])  # 2D obs
            action_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])  # 2D action

            policy = ContinuousActorCriticLayer(obs_space, action_space; hidden_dims = [4, 4], critic_type = QCritic())
            ps = Lux.initialparameters(rng, policy)
            st = Lux.initialstates(rng, policy)

            # Test edge cases that might cause issues
            for batch_size in [1, 2, 3, 4, 5, 8, 16]
                feats = randn(Float32, 2, batch_size)
                actions, new_st = @test_nowarn DRiL.get_actions_from_features(policy, feats, ps, st)
                @test size(actions) == (2, batch_size)
            end
        end

        # Test 5: Network architecture edge cases
        @testset "Network architecture edge cases" begin
            obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
            action_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

            # Test different hidden layer configurations that might cause issues
            hidden_configs = [
                [2, 2],     # Same as input/output
                [4, 2],     # Expand then contract
                [2, 4],     # Contract then expand
                [8, 4, 2],  # Multiple layers
                [16, 8],     # Large hidden layers
            ]

            for hidden_dims in hidden_configs
                @testset "Hidden dims: $(hidden_dims)" begin
                    policy = ContinuousActorCriticLayer(
                        obs_space, action_space;
                        hidden_dims = hidden_dims,
                        critic_type = QCritic()
                    )
                    ps = Lux.initialparameters(rng, policy)
                    st = Lux.initialstates(rng, policy)

                    batch_size = 2
                    feats = randn(Float32, 2, batch_size)
                    actions, new_st = @test_nowarn DRiL.get_actions_from_features(policy, feats, ps, st)
                    @test size(actions) == (2, batch_size)
                end
            end
        end

        # Test 6: Exact reproduction of original error scenario
        @testset "Original error reproduction" begin
            # Based on the stack trace, the error was:
            # "DimensionMismatch: new dimensions (2, 4) must be consistent with array size 4"
            # This happens when trying to reshape 4 elements into (2, 4) = 8 elements

            # Let's test the exact scenario that was causing the issue
            obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])  # 2D obs from SimpleRewardEnv
            action_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])  # 2D action from SimpleRewardEnv

            policy = ContinuousActorCriticLayer(obs_space, action_space; hidden_dims = [64, 64], critic_type = QCritic())
            ps = Lux.initialparameters(rng, policy)
            st = Lux.initialstates(rng, policy)

            # Test the exact batch size from the error (the error occurred with parallel envs)
            batch_size = 4  # BroadcastedParallelEnv scenario
            feats = randn(Float32, 2, batch_size)  # 2D features for 2D obs space

            # This should work without the reshape error
            actions, new_st = @test_nowarn DRiL.get_actions_from_features(policy, feats, ps, st)
            @test size(actions) == (2, batch_size)

            # Also test potential problematic feature dimensions that could trigger the reshape error
            # If the feature extractor somehow outputs wrong dimensions, it would cause the error

            # Test with 1D features (wrong) - this should fail gracefully
            wrong_feats_1d = randn(Float32, 1, batch_size)
            @test_throws Exception DRiL.get_actions_from_features(policy, wrong_feats_1d, ps, st)

            # Test with 4D features (wrong) - this should fail gracefully
            wrong_feats_4d = randn(Float32, 4, batch_size)
            @test_throws Exception DRiL.get_actions_from_features(policy, wrong_feats_4d, ps, st)
        end
    end
end

@testitem "SAC end-to-end gradient computation with real rollouts" tags = [:sac, :integration, :gradients] setup = [SharedTestSetup] begin
    using Random
    using Lux
    using Zygote
    # using ComponentArrays
    using LinearAlgebra
    using DRiL: nested_all_zero, nested_norm

    # Test the complete SAC gradient computation pipeline using real environment data
    # This mimics exactly what happens in the learn! function

    @testset "Real rollout gradient computation" begin
        rng = Random.Xoshiro(42)

        # Create simple test environment and policy
        env = BroadcastedParallelEnv([SharedTestSetup.CustomEnv(8) for _ in 1:2])
        obs_space = DRiL.observation_space(env)
        action_space = DRiL.action_space(env)

        # Create SAC agent and algorithm
        layer = ContinuousActorCriticLayer(
            obs_space, action_space; hidden_dims = [8, 4],
            critic_type = QCritic()
        )
        alg = SAC(; start_steps = 4, batch_size = 4)
        agent = Agent(layer, alg; rng = rng, verbose = 0)

        # Create replay buffer and collect some rollouts
        replay_buffer = ReplayBuffer(obs_space, action_space, 100)

        # Collect initial rollouts (like in learn!)
        n_steps = 4
        fps, _ = DRiL.collect_rollout!(replay_buffer, agent, alg, env, n_steps)

        @test fps > 0  # Sanity check that collection worked
        @test length(replay_buffer) > 0  # Buffer should have data

        # Get real batch data from replay buffer (like in learn!)
        data_loader = DRiL.get_data_loader(replay_buffer, alg.batch_size, 1, true, true, rng)
        batch_data = first(data_loader)

        # Test entropy coefficient loss with real data
        @testset "Entropy coefficient gradient with real data" begin
            if alg.ent_coef isa AutoEntropyCoefficient
                ent_train_state = agent.aux.ent_train_state
                target_entropy = DRiL.get_target_entropy(alg.ent_coef, action_space)

                ent_data = (
                    observations = batch_data.observations,
                    layer_ps = agent.train_state.parameters,
                    layer_st = agent.train_state.states,
                    target_entropy = target_entropy,
                    target_ps = agent.aux.Q_target_parameters,
                    target_st = agent.aux.Q_target_states,
                )

                # Compute gradients exactly like in learn!
                ent_grad, ent_loss, _, ent_train_state = Lux.Training.compute_gradients(
                    AutoZygote(),
                    (model, ps, st, data) -> DRiL.sac_ent_coef_loss(alg, layer, ps, st, data),
                    ent_data,
                    ent_train_state
                )

                # Verify gradient computation succeeded
                @test !isnothing(ent_grad)
                @test haskey(ent_grad, :log_ent_coef)
                @test !iszero(ent_grad.log_ent_coef[1])
                @test isfinite(ent_loss)
                @test ent_loss isa Float32

                # Verify gradient magnitude is reasonable
                grad_magnitude = abs(ent_grad.log_ent_coef[1])
                @test 1.0e-6 < grad_magnitude < 100.0  # Reasonable range
            end
        end

        # Test critic loss with real data
        @testset "Critic gradient with real data" begin
            train_state = agent.train_state

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

            # Compute gradients exactly like in learn!
            critic_grad, critic_loss, critic_stats, train_state = Lux.Training.compute_gradients(
                AutoZygote(),
                (model, ps, st, data) -> DRiL.sac_critic_loss(alg, layer, ps, st, data),
                critic_data,
                train_state
            )

            # Verify gradient computation succeeded
            @test !isnothing(critic_grad)
            @test haskey(critic_grad, :critic_head)
            @test !nested_all_zero(critic_grad.critic_head)
            @test isfinite(critic_loss)
            @test critic_loss isa Float32
            @test critic_loss > 0  # MSE loss should be positive


            # Verify gradient magnitudes are reasonable
            critic_grad_norm = nested_norm(critic_grad.critic_head, Float32)
            actor_grad_norm = nested_norm(critic_grad.actor_head, Float32)
            @test actor_grad_norm < 1.0f-10
            @test critic_grad_norm > 1.0f-10  # Should have meaningful gradients
            @test critic_grad_norm < 1000.0  # But not exploding
        end

        # Test actor loss with real data
        @testset "Actor gradient with real data" begin
            train_state = agent.train_state

            actor_data = (
                observations = batch_data.observations,
                actions = batch_data.actions,
                rewards = batch_data.rewards,
                terminated = batch_data.terminated,
                truncated = batch_data.truncated,
                next_observations = batch_data.next_observations,
                log_ent_coef = agent.aux.ent_train_state.parameters,
            )

            # Compute gradients exactly like in learn!
            actor_grad, actor_loss, _, train_state = Lux.Training.compute_gradients(
                AutoZygote(),
                (model, ps, st, data) -> DRiL.sac_actor_loss(alg, layer, ps, st, data),
                actor_data,
                train_state
            )
            DRiL.zero_critic_grads!(actor_grad, layer)

            # Verify gradient computation succeeded
            @test !isnothing(actor_grad)
            @test haskey(actor_grad, :actor_head)
            @test !nested_all_zero(actor_grad.actor_head)
            @test nested_all_zero(actor_grad.critic_head)
            @test isfinite(actor_loss)
            @test actor_loss isa Float32

            # Actor loss can be positive or negative (depends on Q-values vs entropy)
            # but should be finite
            @test isfinite(actor_loss)

            # Verify gradient magnitudes are reasonable
            actor_grad_norm = nested_norm(actor_grad.actor_head, Float32)
            critic_grad_norm = nested_norm(actor_grad.critic_head, Float32)
            @test actor_grad_norm > 1.0f-10  # Should have meaningful gradients
            @test actor_grad_norm < 1000.0  # But not exploding
            @test critic_grad_norm < 1.0f-10
        end
    end
end
