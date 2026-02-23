# Test module containing shared environments and policies for testing.
# This module will be evaluated once per Julia test process and made available to test items.
@testmodule SharedTestSetup begin
    using Drill
    using Random
    using Drill.Lux

    # Custom environment that gives a reward of 1.0 only at the final timestep of an episode.
    # Equivalent to CustomEnv in stable-baselines3 test_gae.py.
    mutable struct CustomEnv <: AbstractEnv
        max_steps::Int
        n_steps::Int
        observation_space::Box
        action_space::Box
        _terminated::Bool
        _truncated::Bool
        _last_reward::Float32
        _info::Dict{String, Any}
        rng::Random.AbstractRNG

        function CustomEnv(max_steps::Int = 8, rng::Random.AbstractRNG = Random.Xoshiro())
            obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
            act_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
            new(max_steps, 0, obs_space, act_space, false, false, 0.0f0, Dict{String, Any}(), rng)
        end
    end

    Drill.observation_space(env::CustomEnv) = env.observation_space
    Drill.action_space(env::CustomEnv) = env.action_space
    Drill.terminated(env::CustomEnv) = env._terminated
    Drill.truncated(env::CustomEnv) = env._truncated
    Drill.get_info(env::CustomEnv) = env._info

    function Drill.reset!(env::CustomEnv)
        env.n_steps = 0
        env._terminated = false
        env._truncated = false
        env._last_reward = 0.0f0
        env._info = Dict{String, Any}()
        return nothing
    end

    function Drill.act!(env::CustomEnv, action::AbstractArray)
        env.n_steps += 1

        # Reward of 1.0 only at the final step, 0.0 otherwise
        reward = (env.n_steps >= env.max_steps) ? 1.0f0 : 0.0f0

        # Episode terminates when max_steps is reached
        env._terminated = env.n_steps >= env.max_steps
        # To simplify GAE computation checks, we do not consider truncation here
        env._truncated = false
        env._last_reward = reward
        env._info = Dict{String, Any}()

        return reward
    end

    function Drill.observe(env::CustomEnv)
        return rand(env.rng, Float32, 2) .* 2.0f0 .- 1.0f0  # Use env's RNG
    end

    # Infinite horizon environment that gives reward of 1.0 at every step and never terminates.
    # Modified from SB3's InfiniteHorizonEnv to use Box instead of discrete space.
    mutable struct InfiniteHorizonEnv <: AbstractEnv
        n_states::Int
        current_state::Float32
        observation_space::Box
        action_space::Box
        _terminated::Bool
        _truncated::Bool
        _last_reward::Float32
        _info::Dict{String, Any}
        rng::Random.AbstractRNG

        function InfiniteHorizonEnv(n_states::Int = 4, rng::Random.AbstractRNG = Random.Xoshiro())
            # Use continuous observation space [0, n_states] to represent the states
            obs_space = Box(Float32[0.0], Float32[Float32(n_states)])
            act_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
            new(n_states, 0.0f0, obs_space, act_space, false, false, 0.0f0, Dict{String, Any}(), rng)
        end
    end

    Drill.observation_space(env::InfiniteHorizonEnv) = env.observation_space
    Drill.action_space(env::InfiniteHorizonEnv) = env.action_space
    Drill.terminated(env::InfiniteHorizonEnv) = env._terminated
    Drill.truncated(env::InfiniteHorizonEnv) = env._truncated
    Drill.get_info(env::InfiniteHorizonEnv) = env._info

    function Drill.reset!(env::InfiniteHorizonEnv)
        env.current_state = 0.0f0
        env._terminated = false
        env._truncated = false
        env._last_reward = 0.0f0
        env._info = Dict{String, Any}()
        return nothing
    end

    function Drill.act!(env::InfiniteHorizonEnv, action::AbstractArray)
        env.current_state = Float32((Int(env.current_state) + 1) % env.n_states)

        # Always gives reward of 1.0
        reward = 1.0f0

        # Never terminates or truncates
        env._terminated = false
        env._truncated = false
        env._last_reward = reward
        env._info = Dict{String, Any}()

        return reward
    end

    function Drill.observe(env::InfiniteHorizonEnv)
        return [env.current_state]
    end

    # Tracking environment where optimal action matches the observed target value.
    mutable struct TrackingTargetEnv <: AbstractEnv
        max_steps::Int
        current_step::Int
        observation_space::Box
        action_space::Box
        current_obs::Float32
        _terminated::Bool
        _truncated::Bool
        rng::Random.AbstractRNG
    end

    function TrackingTargetEnv(
            max_steps::Int = 16,
            rng::Random.AbstractRNG = Random.Xoshiro()
        )
        obs_space = Box(Float32[0.0], Float32[1.0])
        act_space = Box(Float32[-1.0], Float32[1.0])
        initial_obs = rand(rng, obs_space)[1]
        return TrackingTargetEnv(
            max_steps, 0, obs_space, act_space,
            Float32(initial_obs), false, false, rng
        )
    end

    Drill.observation_space(env::TrackingTargetEnv) = env.observation_space
    Drill.action_space(env::TrackingTargetEnv) = env.action_space
    Drill.terminated(env::TrackingTargetEnv) = env._terminated
    Drill.truncated(env::TrackingTargetEnv) = env._truncated
    Drill.get_info(::TrackingTargetEnv) = Dict{String, Any}()

    function Drill.reset!(env::TrackingTargetEnv)
        env.current_step = 0
        env._terminated = false
        env._truncated = false
        env.current_obs = rand(env.rng, env.observation_space)[1]
        return nothing
    end

    function Drill.act!(env::TrackingTargetEnv, action::AbstractArray)
        env.current_step += 1
        action_low = env.action_space.low[1]
        action_high = env.action_space.high[1]
        action_val = clamp(Float32(action[1]), action_low, action_high)
        diff = abs(action_val - env.current_obs)
        reward = clamp(1.0f0 - diff, 0.0f0, 1.0f0)

        env._terminated = env.current_step >= env.max_steps
        env._truncated = false
        env.current_obs = rand(env.rng, env.observation_space)[1]
        return reward
    end

    function Drill.observe(env::TrackingTargetEnv)
        return Float32[env.current_obs]
    end

    # Simple environment that gives a reward of 1.0 only at the final timestep of an episode.
    # This allows for analytical computation of GAE values for testing.
    mutable struct SimpleRewardEnv <: AbstractEnv
        max_steps::Int
        current_step::Int
        observation_space::Box
        action_space::Box
        _terminated::Bool
        _truncated::Bool
        _last_reward::Float32
        _info::Dict{String, Any}
        rng::Random.AbstractRNG

        function SimpleRewardEnv(max_steps::Int = 8, rng::Random.AbstractRNG = Random.Xoshiro())
            obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
            act_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
            new(max_steps, 0, obs_space, act_space, false, false, 0.0f0, Dict{String, Any}(), rng)
        end
    end

    Drill.observation_space(env::SimpleRewardEnv) = env.observation_space
    Drill.action_space(env::SimpleRewardEnv) = env.action_space
    Drill.terminated(env::SimpleRewardEnv) = env._terminated
    Drill.truncated(env::SimpleRewardEnv) = env._truncated
    Drill.get_info(env::SimpleRewardEnv) = env._info

    function Drill.reset!(env::SimpleRewardEnv)
        env.current_step = 0
        env._terminated = false
        env._truncated = false
        env._last_reward = 0.0f0
        env._info = Dict{String, Any}()
        return nothing
    end

    function Drill.act!(env::SimpleRewardEnv, action::AbstractArray)
        env.current_step += 1

        # Reward of 1.0 only at the final step, 0.0 otherwise
        reward = (env.current_step >= env.max_steps) ? 1.0f0 : 0.0f0

        # Episode terminates when max_steps is reached
        env._terminated = env.current_step >= env.max_steps
        env._truncated = false
        env._last_reward = reward
        env._info = Dict{String, Any}()

        return reward
    end

    function Drill.observe(env::SimpleRewardEnv)
        return rand(env.rng, Float32, 2) .* 2.0f0 .- 1.0f0  # Use env's RNG
    end

    # Custom policy that returns constant values for predictable GAE testing.
    struct ConstantValuePolicy <: Drill.AbstractActorCriticLayer
        observation_space::Box{Float32}
        action_space::Box{Float32}
        constant_value::Float32
    end

    # Implement Lux interface functions
    function Lux.initialparameters(rng::AbstractRNG, policy::ConstantValuePolicy)
        # No learnable parameters needed - the constant value is just configuration
        return NamedTuple()
    end

    function Lux.initialstates(rng::AbstractRNG, policy::ConstantValuePolicy)
        # No states needed for constant policy
        return NamedTuple()
    end

    function Lux.parameterlength(policy::ConstantValuePolicy)
        # No parameters
        return 0
    end

    function Lux.statelength(policy::ConstantValuePolicy)
        # No states
        return 0
    end

    function Drill.predict_values(policy::ConstantValuePolicy, observations::AbstractArray)
        batch_size = size(observations)[end]
        return fill(policy.constant_value, batch_size)
    end

    # Generic predict_values method that takes policy, observations, parameters, and states
    function Drill.predict_values(policy::ConstantValuePolicy, observations::AbstractArray, ps, st)
        batch_size = size(observations)[end]
        return fill(policy.constant_value, batch_size), st
    end

    # Implement the main policy call function
    function (policy::ConstantValuePolicy)(obs::AbstractArray, ps, st; rng::AbstractRNG = Random.default_rng())
        batch_size = size(obs)[end]
        # Random actions in action space bounds
        actions = rand(rng, action_space(policy), batch_size)
        values = fill(policy.constant_value, batch_size)
        logprobs = fill(0.0f0, batch_size)
        return actions, values, logprobs, st
    end

    # Implement predict function
    function Drill.predict_actions(policy::ConstantValuePolicy, obs::AbstractArray, ps, st; deterministic::Bool = false, rng::AbstractRNG = Random.default_rng())
        batch_size = size(obs)[end]
        actions = rand(rng, action_space(policy), batch_size)
        return actions, st
    end

    # Implement evaluate_actions function
    function Drill.evaluate_actions(policy::ConstantValuePolicy, obs::AbstractArray, actions::AbstractArray, ps, st)
        batch_size = size(obs)[end]
        values = fill(policy.constant_value, batch_size)
        logprobs = fill(0.0f0, batch_size)
        entropy = fill(0.0f0, batch_size)
        return values, logprobs, entropy, st
    end

    # Helper function to compute expected GAE advantages analytically.
    function compute_expected_gae(
            rewards::Vector{T}, values::Vector{T}, gamma::T, gae_lambda::T;
            is_terminated::Bool = true, bootstrap_value::Union{Nothing, T} = nothing
        ) where {T <: AbstractFloat}
        n = length(rewards)
        expected_advantages = zeros(T, n)

        # Last step calculation
        if is_terminated || isnothing(bootstrap_value)
            # No bootstrapping for terminated episodes
            expected_advantages[n] = rewards[n] - values[n]
        else
            # Bootstrap for truncated or rollout-limited trajectories
            expected_advantages[n] = rewards[n] + gamma * bootstrap_value - values[n]
        end

        # Backward pass through earlier steps
        for t in (n - 1):-1:1
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            expected_advantages[t] = delta + gamma * gae_lambda * expected_advantages[t + 1]
        end

        return expected_advantages
    end

    # Define AbstractEnvWrapper since it's not in the main codebase
    abstract type AbstractEnvWrapper <: AbstractEnv end

    # Environment wrapper to ensure consistent observations for testing.
    mutable struct ConstantObsWrapper <: AbstractEnvWrapper
        env::AbstractEnv
        constant_obs::Vector{Float32}

        function ConstantObsWrapper(env::AbstractEnv, obs::Vector{Float32})
            new(env, obs)
        end
    end

    # Forward all methods to wrapped environment
    Drill.observation_space(wrapper::ConstantObsWrapper) = Drill.observation_space(wrapper.env)
    Drill.action_space(wrapper::ConstantObsWrapper) = Drill.action_space(wrapper.env)
    Drill.terminated(wrapper::ConstantObsWrapper) = Drill.terminated(wrapper.env)
    Drill.truncated(wrapper::ConstantObsWrapper) = Drill.truncated(wrapper.env)
    Drill.get_info(wrapper::ConstantObsWrapper) = Drill.get_info(wrapper.env)

    function Drill.reset!(wrapper::ConstantObsWrapper)
        Drill.reset!(wrapper.env)
        return nothing
    end

    function Drill.act!(wrapper::ConstantObsWrapper, action::AbstractArray)
        return Drill.act!(wrapper.env, action)
    end

    function Drill.observe(wrapper::ConstantObsWrapper)
        return copy(wrapper.constant_obs)
    end


    struct CustomShapedBoxEnv <: AbstractEnv
        shape::Tuple{Int, Vararg{Int}}
    end
    Drill.reset!(env::CustomShapedBoxEnv) = nothing
    Drill.act!(env::CustomShapedBoxEnv, action::AbstractArray) = rand(Float32)
    Drill.observe(env::CustomShapedBoxEnv) = randn(Float32, env.shape...)
    Drill.observation_space(env::CustomShapedBoxEnv) = Box(Float32[-1.0], Float32[1.0], env.shape)
    Drill.action_space(env::CustomShapedBoxEnv) = Box(Float32[-1.0], Float32[1.0], env.shape)
    Drill.terminated(env::CustomShapedBoxEnv) = false
    Drill.truncated(env::CustomShapedBoxEnv) = false
    Drill.get_info(env::CustomShapedBoxEnv) = Dict{String, Any}()

    struct RandomDiscreteEnv <: AbstractEnv
        obs_space::Box
        act_space::Discrete
    end
    Drill.reset!(env::RandomDiscreteEnv) = nothing
    Drill.act!(env::RandomDiscreteEnv, action::AbstractArray) = randn(Float32)
    Drill.observe(env::RandomDiscreteEnv) = rand(env.obs_space)
    Drill.observation_space(env::RandomDiscreteEnv) = env.obs_space
    Drill.action_space(env::RandomDiscreteEnv) = env.act_space
    Drill.terminated(env::RandomDiscreteEnv) = false
    Drill.truncated(env::RandomDiscreteEnv) = false
    Drill.get_info(env::RandomDiscreteEnv) = Dict{String, Any}()
end
