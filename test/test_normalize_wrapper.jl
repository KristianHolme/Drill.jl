using TestItems

@testitem "RunningMeanStd basic functionality" tags = [:normalization, :running_stats] begin
    using Drill: RunningMeanStd, update!, update_from_moments!
    using Statistics
    using Random

    # Test initialization
    rms = RunningMeanStd{Float32}((3,))
    @test size(rms.mean) == (3,)
    @test size(rms.var) == (3,)
    @test rms.count == 0
    @test all(rms.mean .== 0.0f0)
    @test all(rms.var .== 1.0f0)

    # Test single batch update
    batch = Float32[1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]  # 3x3 batch
    update!(rms, batch)

    @test rms.count == 3
    expected_mean = mean(batch, dims = 2)[:, 1]
    expected_var = var(batch, dims = 2, corrected = false)[:, 1]
    @test all(abs.(rms.mean .- expected_mean) .< 1.0e-6)
    @test all(abs.(rms.var .- expected_var) .< 1.0e-6)

    # Test multiple updates
    batch2 = Float32[0.0 1.0 2.0; 3.0 4.0 5.0; 6.0 7.0 8.0]
    old_count = rms.count
    update!(rms, batch2)

    @test rms.count == old_count + 3
    # Verify the running average is computed correctly
    combined_batch = hcat(batch, batch2)
    combined_mean = mean(combined_batch, dims = 2)[:, 1]
    combined_var = var(combined_batch, dims = 2, corrected = false)[:, 1]
    @test all(abs.(rms.mean .- combined_mean) .< 1.0e-5)
    @test all(abs.(rms.var .- combined_var) .< 1.0e-5)
end

@testitem "RunningMeanStd edge cases" tags = [:normalization, :running_stats, :edge_cases] begin
    using Drill: RunningMeanStd, update!
    using Statistics

    # Test with zero variance
    rms = RunningMeanStd{Float32}((2,))
    constant_batch = Float32[5.0 5.0 5.0; 3.0 3.0 3.0]
    update!(rms, constant_batch)

    @test rms.count == 3
    @test rms.mean ≈ [5.0f0, 3.0f0]
    @test all(rms.var .< 1.0e-6)  # Should be nearly zero

    # Test with single sample
    rms_single = RunningMeanStd{Float32}((1,))
    single_batch = reshape(Float32[42.0], 1, 1)
    update!(rms_single, single_batch)

    @test rms_single.count == 1
    @test rms_single.mean[1] ≈ 42.0f0
    @test rms_single.var[1] ≈ 0.0f0

    # Test scalar case (empty shape)
    rms_scalar = RunningMeanStd{Float32}(())
    scalar_batch = reshape(Float32[1.0, 2.0, 3.0], 1, 3)
    update!(rms_scalar, scalar_batch)

    @test rms_scalar.count == 3
    @test rms_scalar.mean[1] ≈ 2.0f0
    @test rms_scalar.var[1] ≈ var([1.0, 2.0, 3.0], corrected = false)
end

@testitem "NormalizeWrapperEnv dummy environment" tags = [:normalization, :environments] setup = [SharedTestSetup] begin
    using Random

    # Define a dummy environment with non-normalized observations and rewards
    struct DummyEnv <: AbstractEnv end
    Drill.observation_space(::DummyEnv) = Box(Float32[50.0, 50.0, 50.0], Float32[60.0, 60.0, 60.0])
    Drill.action_space(::DummyEnv) = Box(Float32[7.0], Float32[9.0])
    Drill.observe(::DummyEnv) = rand(observation_space(DummyEnv()))
    Drill.terminated(::DummyEnv) = false
    Drill.truncated(::DummyEnv) = false
    Drill.act!(::DummyEnv, action) = randn() * 20.0f0 + 1000.0f0
    Drill.get_info(::DummyEnv) = Dict()
    Drill.reset!(::DummyEnv) = nothing

    # Create parallel environment
    base_env = MultiThreadedParallelEnv([DummyEnv() for _ in 1:4])

    # Test basic wrapper creation
    norm_env = NormalizeWrapperEnv(base_env)

    @test norm_env isa NormalizeWrapperEnv
    @test isequal(observation_space(norm_env), observation_space(base_env))
    @test isequal(action_space(norm_env), action_space(base_env))
    @test number_of_envs(norm_env) == 4
    @test norm_env.training == true
    @test norm_env.norm_obs == true
    @test norm_env.norm_reward == true
end

@testitem "NormalizeWrapperEnv configuration options" tags = [:normalization, :environments] setup = [SharedTestSetup] begin
    # Define the same dummy environment
    struct DummyEnv2 <: AbstractEnv end
    Drill.observation_space(::DummyEnv2) = Box(Float32[0.0, 0.0], Float32[1.0, 1.0])
    Drill.action_space(::DummyEnv2) = Box(Float32[-1.0], Float32[1.0])
    Drill.observe(::DummyEnv2) = rand(observation_space(DummyEnv2()))
    Drill.terminated(::DummyEnv2) = false
    Drill.truncated(::DummyEnv2) = false
    Drill.act!(::DummyEnv2, action) = 5.0f0
    Drill.get_info(::DummyEnv2) = Dict()
    Drill.reset!(::DummyEnv2) = nothing

    base_env = MultiThreadedParallelEnv([DummyEnv2() for _ in 1:2])

    # Test custom configuration
    norm_env = NormalizeWrapperEnv(
        base_env;
        training = false,
        norm_obs = false,
        norm_reward = true,
        clip_obs = 5.0f0,
        clip_reward = 3.0f0,
        gamma = 0.95f0,
        epsilon = 1.0f-4
    )

    @test norm_env.training == false
    @test norm_env.norm_obs == false
    @test norm_env.norm_reward == true
    @test norm_env.clip_obs == 5.0f0
    @test norm_env.clip_reward == 3.0f0
    @test norm_env.gamma == 0.95f0
    @test norm_env.epsilon == 1.0f-4

    # Test training mode control
    @test is_training(norm_env) == false
    norm_env = set_training(norm_env, true)
    @test is_training(norm_env) == true
end

@testitem "NormalizeWrapperEnv observation normalization" tags = [:normalization, :environments] setup = [SharedTestSetup] begin
    using Statistics
    using Random

    # Define deterministic environment for testing
    mutable struct DetEnv <: AbstractEnv
        obs_values::Vector{Vector{Float32}}
        step_count::Int
    end
    DetEnv() = DetEnv([Float32[50, 55, 60], Float32[51, 56, 61], Float32[52, 57, 62]], 0)

    Drill.observation_space(::DetEnv) = Box(Float32[40.0, 40.0, 40.0], Float32[70.0, 70.0, 70.0])
    Drill.action_space(::DetEnv) = Box(Float32[-1.0], Float32[1.0])
    Drill.observe(env::DetEnv) = env.obs_values[min(env.step_count + 1, length(env.obs_values))]
    Drill.terminated(::DetEnv) = false
    Drill.truncated(::DetEnv) = false
    function Drill.act!(env::DetEnv, action)
        env.step_count += 1
        return 0.0f0
    end
    Drill.get_info(::DetEnv) = Dict()
    function Drill.reset!(env::DetEnv)
        env.step_count = 0
        nothing
    end

    # Create environments with different deterministic patterns
    envs = [DetEnv() for _ in 1:2]
    # Offset the second environment's observations
    envs[2].obs_values = [v .+ 10.0f0 for v in envs[2].obs_values]

    base_env = MultiThreadedParallelEnv(envs)
    norm_env = NormalizeWrapperEnv(base_env; training = true, norm_obs = true, norm_reward = false)

    # Reset and collect observations
    reset!(norm_env)

    # Collect data for normalization
    for i in 1:6  # Multiple steps to build statistics
        actions = [rand(action_space(norm_env)) for _ in 1:2]
        rewards = act!(norm_env, actions)
    end

    # Check that observations are being normalized
    final_obs = observe(norm_env)[1]
    final_original = get_original_obs(norm_env)[1]

    # The normalized observations should be different from original
    @test final_obs != final_original

    # Test unnormalization
    unnorm_obs = copy(final_obs)
    unnormalize_obs!(unnorm_obs, norm_env)
    @test all(abs.(unnorm_obs .- final_original) .< 1.0e-5)
end

@testitem "NormalizeWrapperEnv reward normalization" tags = [:normalization, :environments] setup = [SharedTestSetup] begin
    using Statistics
    using Random

    # Environment with high-variance rewards
    mutable struct HighVarRewardEnv <: AbstractEnv
        reward_values::Vector{Float32}
        step_count::Int
    end
    HighVarRewardEnv() = HighVarRewardEnv(Float32[1000, 2000, 500, 1500], 0)

    Drill.observation_space(::HighVarRewardEnv) = Box(Float32[0.0], Float32[1.0])
    Drill.action_space(::HighVarRewardEnv) = Box(Float32[-1.0], Float32[1.0])
    Drill.observe(::HighVarRewardEnv) = Float32[0.5]
    Drill.terminated(::HighVarRewardEnv) = false
    Drill.truncated(::HighVarRewardEnv) = false
    function Drill.act!(env::HighVarRewardEnv, action)
        env.step_count = (env.step_count % length(env.reward_values)) + 1
        return env.reward_values[env.step_count]
    end
    Drill.get_info(::HighVarRewardEnv) = Dict()
    Drill.reset!(env::HighVarRewardEnv) = (env.step_count = 0; nothing)

    base_env = MultiThreadedParallelEnv([HighVarRewardEnv() for _ in 1:2])
    norm_env = NormalizeWrapperEnv(base_env; training = true, norm_obs = false, norm_reward = true)

    reset!(norm_env)
    all_rewards = Float32[]
    original_rewards = Float32[]

    # Collect rewards
    for i in 1:8
        actions = [rand(Float32) for _ in 1:2]
        rewards, _ = act!(norm_env, actions)
        push!(all_rewards, rewards...)
        push!(original_rewards, get_original_rewards(norm_env)...)
    end

    # Check that rewards are being normalized (should have lower variance)
    norm_reward_std = std(all_rewards)
    orig_reward_std = std(original_rewards)

    @test norm_reward_std < orig_reward_std

    # Test unnormalization
    actions = [rand(Float32) for _ in 1:2]
    last_rewards, _ = act!(norm_env, actions)
    last_original = get_original_rewards(norm_env)
    unnorm_rewards = copy(last_rewards)
    unnormalize_rewards!(unnorm_rewards, norm_env)

    @test all(abs.(unnorm_rewards .- last_original) .< 1.0e-4)
end

@testitem "NormalizeWrapperEnv clipping behavior" tags = [:normalization, :environments, :clipping] setup = [SharedTestSetup] begin
    # Environment that produces extreme observations
    struct ExtremeObsEnv <: AbstractEnv end
    Drill.observation_space(::ExtremeObsEnv) = Box(Float32[-1000.0, -1000.0], Float32[1000.0, 1000.0])
    Drill.action_space(::ExtremeObsEnv) = Box(Float32[-1.0], Float32[1.0])
    Drill.observe(::ExtremeObsEnv) = Float32[1000.0, -1000.0]  # Extreme values
    Drill.terminated(::ExtremeObsEnv) = false
    Drill.truncated(::ExtremeObsEnv) = false
    Drill.act!(::ExtremeObsEnv, action) = 0.0f0
    Drill.get_info(::ExtremeObsEnv) = Dict()
    Drill.reset!(::ExtremeObsEnv) = nothing

    base_env = MultiThreadedParallelEnv([ExtremeObsEnv() for _ in 1:1])

    # Test with small clipping bounds
    norm_env = NormalizeWrapperEnv(base_env; clip_obs = 2.0f0, training = true)

    reset!(norm_env)
    # Build some statistics first
    for i in 1:5
        actions = [rand(Float32)]
        act!(norm_env, actions)
    end

    obs = observe(norm_env)[1]

    # All normalized observations should be within clipping bounds
    @test all(abs.(obs) .<= norm_env.clip_obs + 1.0e-6)
end

@testitem "NormalizeWrapperEnv training vs evaluation mode" tags = [:normalization, :environments, :modes] setup = [SharedTestSetup] begin
    using Statistics

    # Simple environment
    mutable struct SimpleEnv <: AbstractEnv
        value::Float32
    end
    SimpleEnv() = SimpleEnv(1.0f0)

    Drill.observation_space(::SimpleEnv) = Box(Float32[0.0], Float32[10.0])
    Drill.action_space(::SimpleEnv) = Box(Float32[-1.0], Float32[1.0])
    Drill.observe(env::SimpleEnv) = Float32[env.value]
    Drill.terminated(::SimpleEnv) = false
    Drill.truncated(::SimpleEnv) = false
    function Drill.act!(env::SimpleEnv, action)
        env.value += 1.0f0
        return env.value
    end
    Drill.get_info(::SimpleEnv) = Dict()
    Drill.reset!(env::SimpleEnv) = (env.value = 1.0f0; nothing)

    base_env = MultiThreadedParallelEnv([SimpleEnv() for _ in 1:2])
    norm_env = NormalizeWrapperEnv(base_env; training = true)

    reset!(norm_env)

    # In training mode, statistics should update
    initial_count = norm_env.obs_rms.count
    actions = [rand(Float32) for _ in 1:2]
    act!(norm_env, actions)
    obs = observe(norm_env) # update stats
    training_count = norm_env.obs_rms.count
    @test training_count > initial_count

    # Switch to evaluation mode
    norm_env = set_training(norm_env, false)
    actions = [rand(Float32) for _ in 1:2]
    act!(norm_env, actions)
    obs = observe(norm_env) # should now not update stats
    eval_count = norm_env.obs_rms.count

    # Statistics should not update in evaluation mode
    @test eval_count == training_count
end

@testitem "NormalizeWrapperEnv terminal observation handling" tags = [:normalization, :environments, :terminal] setup = [SharedTestSetup] begin
    # Environment that terminates and provides terminal observation
    mutable struct TerminalEnv <: AbstractEnv
        steps::Int
        max_steps::Int
    end
    TerminalEnv(max_steps = 3) = TerminalEnv(0, max_steps)

    Drill.observation_space(::TerminalEnv) = Box(Float32[0.0], Float32[10.0])
    Drill.action_space(::TerminalEnv) = Box(Float32[-1.0], Float32[1.0])
    Drill.observe(env::TerminalEnv) = Float32[env.steps]
    Drill.terminated(env::TerminalEnv) = env.steps >= env.max_steps
    Drill.truncated(::TerminalEnv) = false
    function Drill.act!(env::TerminalEnv, action)
        env.steps += 1
        return Float32(env.steps)
    end
    Drill.get_info(::TerminalEnv) = Dict{String, Any}()
    function Drill.reset!(env::TerminalEnv)
        env.steps = 0
        nothing
    end


    base_env = BroadcastedParallelEnv([TerminalEnv(2) for _ in 1:2])
    norm_env = NormalizeWrapperEnv(base_env; training = true)

    reset!(norm_env)

    # Step until termination
    actions = [[rand(Float32)] for _ in 1:2]
    rewards, terms, truncs, infos = act!(norm_env, actions)
    obs = observe(norm_env)
    @test !any(terms)

    actions = [[rand(Float32)] for _ in 1:2]
    rewards, terms, truncs, infos = act!(norm_env, actions)
    obs = observe(norm_env)
    @test all(terms)

    # Check that terminal observations are normalized in info
    for info in infos
        if haskey(info, "terminal_observation")
            terminal_obs = info["terminal_observation"]
            @test terminal_obs isa Vector{Float32}
            @test length(terminal_obs) == 1
            # Terminal observation should be normalized (different from raw value)
            # The raw terminal observation would be [2f0], normalized should be different
            @test abs(terminal_obs[1]) ≤ 2.0f0  # Should be normalized
        end
    end
end

@testitem "NormalizeWrapperEnv interface compliance" tags = [:normalization, :environments, :interface] setup = [SharedTestSetup] begin
    using Random

    # Simple test environment
    struct SimpleTestEnv <: AbstractEnv
        rng::Random.AbstractRNG
    end
    Drill.observation_space(::SimpleTestEnv) = Box(Float32[0.0, 0.0], Float32[1.0, 1.0])
    Drill.action_space(::SimpleTestEnv) = Box(Float32[-1.0], Float32[1.0])
    Drill.observe(env::SimpleTestEnv) = rand(env.rng, Float32, 2)
    Drill.terminated(::SimpleTestEnv) = false
    Drill.truncated(::SimpleTestEnv) = false
    Drill.act!(env::SimpleTestEnv, action) = rand(env.rng, Float32)
    Drill.get_info(::SimpleTestEnv) = Dict()
    Drill.reset!(::SimpleTestEnv) = nothing

    base_env = MultiThreadedParallelEnv([SimpleTestEnv(Random.MersenneTwister(i)) for i in 1:3])
    norm_env = NormalizeWrapperEnv(base_env)

    # Test that all required interface methods exist and work
    @test hasmethod(observation_space, (typeof(norm_env),))
    @test hasmethod(action_space, (typeof(norm_env),))
    @test hasmethod(number_of_envs, (typeof(norm_env),))
    @test hasmethod(reset!, (typeof(norm_env),))
    @test hasmethod(observe, (typeof(norm_env),))
    @test hasmethod(act!, (typeof(norm_env), Vector))
    @test hasmethod(terminated, (typeof(norm_env),))
    @test hasmethod(truncated, (typeof(norm_env),))
    @test hasmethod(get_info, (typeof(norm_env),))

    # Test basic functionality
    obs_space = observation_space(norm_env)
    act_space = action_space(norm_env)
    n_envs = number_of_envs(norm_env)

    @test obs_space isa Box{Float32}
    @test act_space isa Box{Float32}
    @test n_envs == 3

    # Test reset and observe
    reset!(norm_env)
    initial_obs = observe(norm_env)
    @test length(initial_obs) == 3
    @test all(obs -> length(obs) == 2, initial_obs)

    current_obs = observe(norm_env)
    @test length(current_obs) == 3
    @test all(obs -> length(obs) == 2, current_obs)

    # Test act!
    actions = rand(action_space(norm_env), 3)
    rewards, terms, truncs, infos = act!(norm_env, actions)

    @test length(rewards) == 3
    @test all(r -> r isa Float32, rewards)

    # Test other methods
    @test length(terms) == 3
    @test length(truncs) == 3
    @test length(infos) == 3
    @test all(t -> t isa Bool, terms)
    @test all(t -> t isa Bool, truncs)
    @test all(i -> i isa Dict, infos)

    # Test seeding
    norm_env = set_training(norm_env, false)
    Random.seed!(norm_env, 42)
    reset!(norm_env)
    obs1 = observe(norm_env)
    Random.seed!(norm_env, 42)
    reset!(norm_env)
    obs2 = observe(norm_env)
    @test all([isapprox(o1, o2) for (o1, o2) in zip(obs1, obs2)])
    # Note: Due to running statistics, perfect reproducibility is not expected
    # but the underlying environment should be seeded
end
