using Test
using Drill
using DrillInterface
using Random
using Lux
using Lux: AutoZygote, AutoEnzyme
using Zygote
using Enzyme
include("setup.jl")
using .TestSetup

# Copies of every number and numeric array in a nested structure (learner, optimizer
# state, parameters), in a fixed order. RNGs are skipped: updates may advance them.
function numeric_leaves(x, out = Any[])
    if x isa AbstractArray{<:Number}
        push!(out, copy(x))
    elseif x isa Number
        push!(out, x)
    elseif x isa AbstractArray || x isa Tuple || x isa NamedTuple
        foreach(y -> numeric_leaves(y, out), x)
    elseif x isa AbstractRNG || x isa Function || x === nothing
        nothing
    elseif isstructtype(typeof(x))
        foreach(f -> numeric_leaves(getfield(x, f), out), fieldnames(typeof(x)))
    end
    return out
end

flat_arrays(x) = reduce(vcat, [vec(Float64.(l)) for l in numeric_leaves(x) if l isa AbstractArray])

function ppo_setup(; n_envs = 2, n_steps = 8, kwargs...)
    env = BroadcastedParallelEnv([CustomEnv(8, Random.Xoshiro(i)) for i in 1:n_envs])
    model = ActorCriticModel(observation_space(env), action_space(env); hidden_dims = [16, 16])
    alg = PPO(; n_steps, batch_size = n_steps * n_envs, epochs = 2, kwargs...)
    cache = init(RLProblem(env, model), alg; max_steps = n_steps * n_envs, verbosity = 0, rng = Random.Xoshiro(42))
    return env, alg, cache
end

function ppo_batch(cache, alg, env)
    Drill.collect_rollout!(cache.buffer, cache, alg, env)
    Drill.prepare_rollout!(cache.buffer, alg)
    b = cache.buffer
    actions = Drill.prepare_training_actions(b.actions, action_space(b))
    return (b.observations, actions, b.advantages, b.returns, b.logprobs, b.values)
end

function sac_setup(; kwargs...)
    env = BroadcastedParallelEnv([CustomEnv(8, Random.Xoshiro(i)) for i in 1:2])
    model = ContinuousActorCriticModel(
        observation_space(env), action_space(env); hidden_dims = [16, 16], critic_type = QCritic(),
    )
    alg = SAC(; start_steps = 4, batch_size = 8, kwargs...)
    cache = init(RLProblem(env, model), alg; max_steps = 64, verbosity = 0, rng = Random.Xoshiro(42))
    Drill.collect_rollout!(cache.buffer, cache, alg, env, 8)
    batch = first(Drill.get_data_loader(cache.buffer, alg.batch_size, 1, true, false, Random.Xoshiro(3)))
    target_entropy = Drill.get_target_entropy(alg.ent_coef, action_space(env))
    return alg, cache, batch, target_entropy
end

@testset "value_and_gradient" begin
    # f(x, a) = sum(a .* x.w .^ 2) + 3 * sum(x.b): ∂w = 2 a .* w, ∂b = 3
    f(x, a) = (sum(a .* x.w .^ 2) + 3 * sum(x.b), (; n = length(x.w), a_sum = sum(a)))
    for (name, ad) in (("Zygote", AutoZygote()), ("Enzyme", AutoEnzyme()))
        @testset "$name" begin
            x = (; w = Float32[1, -2, 3], b = Float32[0.5, 0.25])
            a = Float32[2, 3, 4]
            x0 = deepcopy(x)
            a0 = copy(a)
            loss, aux, grad = Drill.value_and_gradient(ad, f, x, a)
            @test loss ≈ 2 * 1 + 3 * 4 + 4 * 9 + 3 * 0.75
            @test aux == (; n = 3, a_sum = 9.0f0)
            @test grad.w ≈ 2 .* a .* x.w
            @test grad.b ≈ Float32[3, 3]
            # inputs are not modified; the constant argument gets no gradient
            @test x == x0
            @test a == a0
            @test keys(grad) == (:w, :b)
        end
    end
    @test_throws ArgumentError Drill.value_and_gradient(:not_an_ad_backend, f, (; w = [1.0f0], b = [1.0f0]), [1.0f0])
end

@testset "global_norm and clip_by_global_norm!" begin
    grads = (; a = Float32[3], b = (; c = Float32[4], d = nothing), e = nothing)
    n = Drill.global_norm(grads)
    @test n ≈ 5
    @test Drill.global_norm((; a = nothing)) == 0

    original = deepcopy(grads)
    unclipped = Drill.clip_by_global_norm!(grads, 10.0f0, n)
    @test unclipped === grads
    @test grads.a ≈ original.a
    @test grads.b.c ≈ original.b.c
    @test grads.b.d === nothing
    @test grads.e === nothing

    Drill.clip_by_global_norm!(grads, 1.0f0, n)
    @test Drill.global_norm(grads) ≈ 1 rtol = 1.0e-5
    @test grads.a ./ grads.b.c ≈ original.a ./ original.b.c
    @test grads.b.d === nothing
end

@testset "ppo_update" begin
    env, alg, cache = ppo_setup()
    batch = ppo_batch(cache, alg, env)
    learner = cache.learner
    @test learner isa PPOLearner
    before = deepcopy(learner)

    new_learner, metrics = ppo_update(alg, cache.model, deepcopy(before), batch, AutoZygote())
    @test new_learner isa PPOLearner
    @test flat_arrays(new_learner.ps) != flat_arrays(before.ps)
    @test numeric_leaves(new_learner.opt_state) != numeric_leaves(before.opt_state)
    @test all(isfinite, values(metrics))
    @test keys(metrics) == (:loss, :policy_loss, :value_loss, :entropy_loss, :approx_kl_div, :clip_fraction, :grad_norm)

    # Same input, same output: the update has no hidden state
    again, metrics_again = ppo_update(alg, cache.model, deepcopy(before), batch, AutoZygote())
    @test numeric_leaves(again) == numeric_leaves(new_learner)
    @test metrics_again == metrics
end

@testset "sac_update" begin
    alg, cache, batch, target_entropy = sac_setup()
    learner = cache.learner
    @test learner isa SACLearner
    before = deepcopy(learner)

    new_learner, metrics = sac_update(alg, cache.model, deepcopy(before), batch, AutoZygote(), target_entropy, Val(true))
    @test new_learner isa SACLearner
    @test flat_arrays(new_learner.actor_ps) != flat_arrays(before.actor_ps)
    @test flat_arrays(new_learner.critic_ps) != flat_arrays(before.critic_ps)
    @test flat_arrays(new_learner.log_ent_coef) != flat_arrays(before.log_ent_coef)
    @test all(isfinite, values(metrics))
    # the reported coefficient is the pre-step value
    @test metrics.entropy_coefficient ≈ Drill.entropy_coefficient(before)
end

struct TargetRecorder <: AbstractCallback
    targets::Vector{Vector{Float64}}
    updates::Vector{Int}
end
TargetRecorder() = TargetRecorder(Vector{Float64}[], Int[])
function record!(r::TargetRecorder, cache)
    push!(r.targets, flat_arrays(cache.learner.target_ps))
    push!(r.updates, cache.gradient_updates)
    return r
end
function Drill.on_rollout_start(r::TargetRecorder, cache)
    record!(r, cache)
    return true
end

@testset "SAC target update interval" begin
    alg, cache, batch, target_entropy = sac_setup(; target_update_interval = 2)
    learner = cache.learner

    # Driven directly: Val(false) keeps the target, Val(true) applies the Polyak update
    kept, _ = sac_update(alg, cache.model, deepcopy(learner), batch, AutoZygote(), target_entropy, Val(false))
    @test numeric_leaves(kept.target_ps) == numeric_leaves(learner.target_ps)
    moved, _ = sac_update(alg, cache.model, deepcopy(learner), batch, AutoZygote(), target_entropy, Val(true))
    @test flat_arrays(moved.target_ps) != flat_arrays(learner.target_ps)
    expected = Drill.polyak!(deepcopy(learner.target_ps), moved.critic_ps, alg.tau)
    @test flat_arrays(moved.target_ps) ≈ flat_arrays(expected)

    # The schedule used by training: gradient step k updates the target iff k % 2 == 0
    current = deepcopy(learner)
    for k in 1:4
        update_target = k % alg.target_update_interval == 0
        previous_target = flat_arrays(current.target_ps)
        current, _ = sac_update(alg, cache.model, current, batch, AutoZygote(), target_entropy, Val(update_target))
        @test (flat_arrays(current.target_ps) != previous_target) == update_target
    end

    # Through training: between rollouts, the target moved iff an even step was taken
    alg2, cache2, _, _ = sac_setup(; target_update_interval = 2, train_freq = 1, gradient_steps = 1)
    recorder = TargetRecorder()
    cache2.callbacks = [recorder]
    solve!(cache2)
    record!(recorder, cache2)
    @test cache2.gradient_updates > 2
    for i in 2:length(recorder.targets)
        n_new = recorder.updates[i] - recorder.updates[i - 1]
        n_new == 0 && continue
        @test n_new == 1
        @test (recorder.targets[i] != recorder.targets[i - 1]) == iseven(recorder.updates[i])
    end
end

@testset "PPO KL early stop discards the update" begin
    # One rollout of 16 steps, 2 minibatches per epoch, 4 epochs
    env = BroadcastedParallelEnv([CustomEnv(8, Random.Xoshiro(i)) for i in 1:2])
    model = ActorCriticModel(observation_space(env), action_space(env); hidden_dims = [16, 16])
    alg = PPO(; n_steps = 8, batch_size = 8, epochs = 4, target_kl = 1.0f-12)
    cache = init(RLProblem(env, model), alg; max_steps = 16, verbosity = 0, rng = Random.Xoshiro(42))
    initial = flat_arrays(Drill.parameters(cache))
    solve!(cache)
    # The first minibatch sees the rollout policy (KL ≈ 0) and may be kept; any later
    # minibatch exceeds the threshold, and its update is discarded.
    @test cache.gradient_updates <= 1
    @test (flat_arrays(Drill.parameters(cache)) == initial) == (cache.gradient_updates == 0)

    reference = init(
        RLProblem(env, model), PPO(; n_steps = 8, batch_size = 8, epochs = 4);
        max_steps = 16, verbosity = 0, rng = Random.Xoshiro(42),
    )
    solve!(reference)
    @test reference.gradient_updates == 8
end

@testset "Learner save/load round trip" begin
    @testset "PPO" begin
        env, alg, cache = ppo_setup()
        solve!(cache)
        @test cache.gradient_updates > 0
        mktempdir() do dir
            path = save_model_params_and_state(cache, joinpath(dir, "ppo"))
            _, _, fresh = ppo_setup()
            @test numeric_leaves(fresh.learner) != numeric_leaves(cache.learner)
            load_model_params_and_state!(fresh, alg, path)
            @test fresh.learner isa PPOLearner
            @test numeric_leaves(fresh.learner) == numeric_leaves(cache.learner)
        end
    end
    @testset "SAC" begin
        alg, cache, _, _ = sac_setup()
        solve!(cache)
        @test cache.gradient_updates > 0
        mktempdir() do dir
            path = save_model_params_and_state(cache, joinpath(dir, "sac"))
            _, fresh, _, _ = sac_setup()
            @test numeric_leaves(fresh.learner) != numeric_leaves(cache.learner)
            load_model_params_and_state!(fresh, alg, path)
            @test fresh.learner isa SACLearner
            @test numeric_leaves(fresh.learner) == numeric_leaves(cache.learner)
        end
    end
end
