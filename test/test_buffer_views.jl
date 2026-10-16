using Test
using Drill
using DrillInterface
using Random
include("setup.jl")
using .TestSetup

# Two envs, three steps. Env 1 is truncated at step 3; env 2 terminates at step 2 and starts
# a new episode at step 3. Reward of env 1 at step t is t, of env 2 is 10 + t.
function two_env_buffer()
    buffer = RolloutBuffer(Box(-10.0f0, 10.0f0, (2,)), Box(-1.0f0, 1.0f0, (1,)), 3, 2)
    for t in 1:3
        obs = reshape(Float32[t, -t, 10 + t, -10 - t] ./ 10, 2, 2)
        store_step!(
            buffer, t, obs, reshape(Float32[t, -t] ./ 10, 1, 2),
            Float32[t, 10 + t], Float32[-t, -10 - t], Float32[2t, 20 + 2t],
            [false, t == 2], [t == 3, false],
        )
    end
    return buffer
end

@testset "transitions(::RolloutBuffer)" begin
    buffer = two_env_buffer()
    trs = transitions(buffer)
    @test length(trs) == length(buffer)
    @test trs.rewards == buffer.rewards
    @test trs.terminateds == buffer.terminateds
    @test propertynames(trs) == (
        :observations, :actions, :rewards, :logprobs, :values, :advantages, :returns,
        :terminateds, :truncateds, :bootstrap_values,
    )

    tr = trs[4]
    @test tr isa NamedTuple
    @test tr.rewards == 12
    @test tr.terminateds
    @test tr.observations == buffer.observations[:, 4]
    @test tr.actions == buffer.actions[:, 4]

    # Views share memory with the buffer
    trs.advantages[2] = 5.0f0
    @test buffer.advantages[2] == 5.0f0
    trs.observations[1][2] = 7.0f0
    @test buffer.observations[2, 1] == 7.0f0
end

@testset "env_transitions" begin
    buffer = two_env_buffer()
    @test env_transitions(buffer, 1).rewards == Float32[1, 2, 3]
    @test env_transitions(buffer, 2).rewards == Float32[11, 12, 13]
    @test env_transitions(buffer, 2).terminateds == [false, true, false]
    @test env_transitions(buffer, 2)[3].observations == Float32[1.3, -1.3]
    @test_throws ArgumentError env_transitions(buffer, 3)
end

@testset "episodes" begin
    buffer = two_env_buffer()
    eps = episodes(buffer)
    @test length(eps) == 3
    @test [ep.rewards for ep in eps] == [Float32[1, 2, 3], Float32[11, 12], Float32[13]]
    # How each episode ended: truncated, terminated, cut by the end of the rollout
    @test [(ep[end].terminateds, ep[end].truncateds) for ep in eps] ==
        [(false, true), (true, false), (false, false)]
    @test sum(sum(ep.rewards) for ep in eps) == sum(buffer.rewards)

    eps[2].values[1] = -1.0f0
    @test buffer.values[2] == -1.0f0
end

@testset "episodes match per-episode GAE" begin
    rng = Xoshiro(3)
    n_steps, n_envs = 9, 3
    gamma, gae_lambda = 0.95f0, 0.9f0
    buffer = RolloutBuffer(Box(-1.0f0, 1.0f0, (2,)), Box(-1.0f0, 1.0f0, (1,)), n_steps, n_envs)
    for t in 1:n_steps
        terminated = rand(rng, n_envs) .< 0.2
        truncated = .!terminated .& (rand(rng, n_envs) .< 0.15)
        store_step!(
            buffer, t, rand(rng, Float32, 2, n_envs), rand(rng, Float32, 1, n_envs),
            randn(rng, Float32, n_envs), zeros(Float32, n_envs), randn(rng, Float32, n_envs),
            terminated, truncated,
        )
    end
    buffer.bootstrap_values .= randn(rng, Float32, length(buffer))
    compute_gae!(buffer, gamma, gae_lambda)

    eps = episodes(buffer)
    @test sum(length, eps) == length(buffer)
    @test length(eps) > n_envs
    for ep in eps
        expected = compute_expected_gae(
            collect(ep.rewards), collect(ep.values), gamma, gae_lambda;
            is_terminated = ep[end].terminateds, bootstrap_value = ep[end].bootstrap_values,
        )
        @test isapprox(ep.advantages, expected; atol = 1.0e-5)
        @test isapprox(ep.returns, ep.advantages .+ ep.values; atol = 1.0e-6)
    end
end

@testset "transitions with matrix observations and discrete actions" begin
    buffer = RolloutBuffer(Box(-1.0f0, 1.0f0, (2, 3)), Discrete(4), 2, 2)
    store_step!(
        buffer, 1, rand(Float32, 2, 3, 2), reshape([1, 4], 1, 2),
        zeros(Float32, 2), zeros(Float32, 2), zeros(Float32, 2), [false, false], [false, false],
    )
    tr = transitions(buffer)[2]
    @test size(tr.observations) == (2, 3)
    @test tr.observations == buffer.observations[:, :, 2]
    @test tr.actions == [4]
end

@testset "transitions(::ReplayBuffer) oldest first" begin
    buffer = ReplayBuffer(Box(-100.0f0, 100.0f0, (2,)), Box(-100.0f0, 100.0f0, (1,)), 5)
    @test isempty(transitions(buffer))

    # Transition k has observation [k, -k], action k, reward k and next observation [k + 1, -k].
    function add_range!(buffer, ks)
        n = length(ks)
        add_transitions!(
            buffer, Float32.(vcat(ks', -ks')), Float32.(reshape(ks, 1, n)), Float32.(ks),
            iseven.(ks), falses(n) |> collect, Float32.(vcat(ks' .+ 1, -ks')),
        )
        return buffer
    end

    add_range!(buffer, 1:3)
    trs = transitions(buffer)
    @test trs.rewards == Float32[1, 2, 3]
    @test trs[2].terminated
    @test trs[3].next_observations == Float32[4, -3]

    # After wrap-around the ring stores 6, 7, 3, 4, 5; the view is in time order
    add_range!(buffer, 4:7)
    trs = transitions(buffer)
    @test trs.rewards == Float32[3, 4, 5, 6, 7]
    @test [tr.observations[1] for tr in trs] == Float32[3, 4, 5, 6, 7]
    @test propertynames(trs) == (:observations, :actions, :rewards, :terminated, :truncated, :next_observations)
end
