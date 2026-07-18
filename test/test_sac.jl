using Test
using Drill
using DrillInterface
using Random
using Lux
using Zygote
using LinearAlgebra
include("setup.jl")
using .TestSetup

@testset "SAC get_actions_from_features dimension mismatch detection" begin
    rng = Random.Xoshiro(42)

    @testset "get_actions_from_features dimension mismatch detection" begin
        @testset "Compatible feature dimensions" begin
            obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
            action_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

            layer = ContinuousActorCriticModel(obs_space, action_space; hidden_dims = [4, 4], critic_type = QCritic())
            ps = Lux.initialparameters(rng, layer)
            st = Lux.initialstates(rng, layer)

            for batch_size in [1, 2, 4, 8]
                obs = rand(obs_space, batch_size)
                batch_obs = DrillInterface.batch(obs, obs_space)
                actor_feats, critic_feats, st = Drill.extract_features(layer, batch_obs, ps, st)
                @test_nowarn Drill.get_actions_from_features(layer, actor_feats, ps, st)
            end
        end

        @testset "Feature dimension mismatches" begin
            @testset "Original error scenario: 2D obs -> 2D action" begin
                obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
                action_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

                layer = ContinuousActorCriticModel(obs_space, action_space; hidden_dims = [4, 4], critic_type = QCritic())
                ps = Lux.initialparameters(rng, layer)
                st = Lux.initialstates(rng, layer)

                batch_size = 2
                feats = randn(Float32, 2, batch_size)
                @test_nowarn Drill.get_actions_from_features(layer, feats, ps, st)
            end

            @testset "Mismatched feature dimensions" begin
                obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
                action_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

                layer = ContinuousActorCriticModel(obs_space, action_space; hidden_dims = [6, 4], critic_type = QCritic())
                ps = Lux.initialparameters(rng, layer)
                st = Lux.initialstates(rng, layer)

                batch_size = 2

                wrong_feats = randn(Float32, 4, batch_size)
                @test_throws Exception Drill.get_actions_from_features(layer, wrong_feats, ps, st)

                wrong_feats_1d = randn(Float32, 1, batch_size)
                @test_throws Exception Drill.get_actions_from_features(layer, wrong_feats_1d, ps, st)
            end
        end

        @testset "Various obs/action dimension combinations" begin
            test_cases = [
                (1, 1), (1, 2), (1, 4),
                (2, 1), (2, 2), (2, 4),
                (4, 1), (4, 2), (4, 4),
                (8, 2), (8, 4),
            ]

            for (obs_dim, action_dim) in test_cases
                @testset "$(obs_dim)D obs, $(action_dim)D action" begin
                    obs_space = Box(Float32[-1.0], Float32[1.0], (obs_dim,))
                    action_space = Box(Float32[-1.0], Float32[1.0], (action_dim,))

                    hidden_dim = max(obs_dim, action_dim) * 2
                    layer = ContinuousActorCriticModel(
                        obs_space, action_space;
                        hidden_dims = [hidden_dim, hidden_dim],
                        critic_type = QCritic()
                    )
                    ps = Lux.initialparameters(rng, layer)
                    st = Lux.initialstates(rng, layer)

                    batch_size = 2
                    feats = randn(Float32, obs_dim, batch_size)
                    actions, new_st = @test_nowarn Drill.get_actions_from_features(layer, feats, ps, st)

                    @test size(actions) == (action_dim, batch_size)
                end
            end
        end

        @testset "Batch size edge cases" begin
            obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
            action_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

            layer = ContinuousActorCriticModel(obs_space, action_space; hidden_dims = [4, 4], critic_type = QCritic())
            ps = Lux.initialparameters(rng, layer)
            st = Lux.initialstates(rng, layer)

            for batch_size in [1, 2, 3, 4, 5, 8, 16]
                feats = randn(Float32, 2, batch_size)
                actions, new_st = @test_nowarn Drill.get_actions_from_features(layer, feats, ps, st)
                @test size(actions) == (2, batch_size)
            end
        end

        @testset "Network architecture edge cases" begin
            obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
            action_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

            hidden_configs = [
                [2, 2],
                [4, 2],
                [2, 4],
                [8, 4, 2],
                [16, 8],
            ]

            for hidden_dims in hidden_configs
                @testset "Hidden dims: $(hidden_dims)" begin
                    layer = ContinuousActorCriticModel(
                        obs_space, action_space;
                        hidden_dims = hidden_dims,
                        critic_type = QCritic()
                    )
                    ps = Lux.initialparameters(rng, layer)
                    st = Lux.initialstates(rng, layer)

                    batch_size = 2
                    feats = randn(Float32, 2, batch_size)
                    actions, new_st = @test_nowarn Drill.get_actions_from_features(layer, feats, ps, st)
                    @test size(actions) == (2, batch_size)
                end
            end
        end

        @testset "Original error reproduction" begin
            obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
            action_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

            layer = ContinuousActorCriticModel(obs_space, action_space; hidden_dims = [64, 64], critic_type = QCritic())
            ps = Lux.initialparameters(rng, layer)
            st = Lux.initialstates(rng, layer)

            batch_size = 4
            feats = randn(Float32, 2, batch_size)

            actions, new_st = @test_nowarn Drill.get_actions_from_features(layer, feats, ps, st)
            @test size(actions) == (2, batch_size)

            wrong_feats_1d = randn(Float32, 1, batch_size)
            @test_throws Exception Drill.get_actions_from_features(layer, wrong_feats_1d, ps, st)

            wrong_feats_4d = randn(Float32, 4, batch_size)
            @test_throws Exception Drill.get_actions_from_features(layer, wrong_feats_4d, ps, st)
        end
    end
end

@testset "SAC end-to-end gradient computation with real rollouts" begin
    using Drill: nested_all_zero, nested_norm

    @testset "Real rollout gradient computation" begin
        rng = Random.Xoshiro(42)

        env = BroadcastedParallelEnv([CustomEnv(8) for _ in 1:2])
        obs_space = DrillInterface.observation_space(env)
        action_space = DrillInterface.action_space(env)

        layer = ContinuousActorCriticModel(
            obs_space, action_space; hidden_dims = [8, 4],
            critic_type = QCritic()
        )
        alg = SAC(; start_steps = 4, batch_size = 4)
        cache = init(RLProblem(env, layer), alg; max_steps = 32, rng, verbosity = 0)

        replay_buffer = ReplayBuffer(obs_space, action_space, 100)

        n_steps = 4
        fps, _ = Drill.collect_rollout!(replay_buffer, cache, alg, env, n_steps)

        @test fps > 0
        @test length(replay_buffer) > 0

        data_loader = Drill.get_data_loader(replay_buffer, alg.batch_size, 1, true, true, rng)
        batch_data = first(data_loader)

        @testset "Critic gradient with real data" begin
            ts = cache.train_state

            target_q_values = Drill.compute_target_q_values(
                alg, layer, Drill.parameters(ts), Drill.states(ts),
                (
                    next_observations = batch_data.next_observations,
                    terminated = batch_data.terminated,
                    log_ent_coef = Drill.entropy_parameters(ts),
                    rewards = batch_data.rewards,
                    target_ps = ts.target_parameters,
                    target_st = ts.target_states,
                );
                rng = rng
            )

            critic_data = (
                observations = batch_data.observations,
                actions = batch_data.actions,
                target_q_values = target_q_values,
                actor_ps = ts.actor_ts.parameters,
                actor_st = ts.actor_ts.states,
            )

            critic_grad, critic_loss, critic_stats, _ = Lux.Training.compute_gradients(
                AutoZygote(),
                Drill.SACCriticObjective(alg, rng),
                critic_data,
                ts.critic_ts,
            )

            @test !isnothing(critic_grad)
            @test haskey(critic_grad, :critic_head)
            @test !haskey(critic_grad, :actor_head)
            @test !nested_all_zero(critic_grad.critic_head)
            @test isfinite(critic_loss)
            @test critic_loss isa Float32
            @test critic_loss > 0

            critic_grad_norm = nested_norm(critic_grad.critic_head, Float32)
            @test critic_grad_norm > 1.0f-10
            @test critic_grad_norm < 1000.0
        end

        @testset "Actor gradient with real data" begin
            ts = cache.train_state

            ent_coef = Float32(Drill.entropy_coefficient(ts))
            actor_data = (
                observations = batch_data.observations,
                ent_coef = ent_coef,
                critic_ps = ts.critic_ts.parameters,
                critic_st = ts.critic_ts.states,
            )

            actor_grad, actor_loss, _, _ = Lux.Training.compute_gradients(
                AutoZygote(),
                Drill.SACActorObjective(alg, rng),
                actor_data,
                ts.actor_ts,
            )

            @test !isnothing(actor_grad)
            @test haskey(actor_grad, :actor_head)
            @test !haskey(actor_grad, :critic_head)
            @test !nested_all_zero(actor_grad.actor_head)
            @test isfinite(actor_loss)
            @test actor_loss isa Float32

            actor_grad_norm = nested_norm(actor_grad.actor_head, Float32)
            @test actor_grad_norm > 1.0f-10
            @test actor_grad_norm < 1000.0
        end
    end
end
