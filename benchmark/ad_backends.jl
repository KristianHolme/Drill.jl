# Compare AD backends and devices: Zygote, Enzyme, Reactant on the CPU and Reactant on a GPU.
#
# Two modes:
#   update     One `ppo_update` or `sac_update` (loss, gradient, optimizer step, metrics
#              to the host) for each hidden width and batch size.
#   iteration  Full PPO iterations (rollout of 8 envs × 256 steps, then 10 epochs of
#              64-sample minibatches) on a synthetic environment.
#
# Usage (one backend per process, as Reactant's backend is chosen once per session):
#   julia --project=<env> benchmark/ad_backends.jl update <backend> <ppo|sac> [widths] [batches]
#   julia --project=<env> benchmark/ad_backends.jl iteration <backend> <width>
# where <backend> is zygote, enzyme, reactant-cpu or reactant-gpu, and widths and batches
# are comma-separated, e.g. 64,256,1024 and 64,1024.
#
# <env> needs Drill and DrillInterface (developed from this repository), Lux, Zygote,
# Enzyme and Reactant:
#   julia --project=<env> -e 'using Pkg; Pkg.develop([PackageSpec(path = "DrillInterface"),
#       PackageSpec(path = ".")]); Pkg.add(["Lux", "Zygote", "Enzyme", "Reactant"])'
#
# On Reactant GPU, a process that compiles updates for several shapes can exit without an
# error message. Run one width and batch size per process there.

using Drill
using DrillInterface
using Lux
using Random
using Statistics
using Printf
using Zygote
using Enzyme
using Reactant

const OBS_DIM = 16
const ACT_DIM = 4

const MODE = ARGS[1]
const BACKEND = ARGS[2]

if startswith(BACKEND, "reactant")
    Reactant.set_default_backend(BACKEND == "reactant-gpu" ? "gpu" : "cpu")
    const DEV = Lux.reactant_device()
    const AD = Drill.gradient_backend(DEV, AutoEnzyme())
elseif BACKEND in ("zygote", "enzyme")
    const DEV = Lux.cpu_device()
    # SAC with Enzyme needs runtime activity; PPO uses the same mode for comparability.
    const AD = BACKEND == "zygote" ? AutoZygote() :
        AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
else
    error("Unknown backend $BACKEND")
end

const OBS_SPACE = Box(-1.0f0, 1.0f0, (OBS_DIM,))
const ACT_SPACE = Box(-1.0f0, 1.0f0, (ACT_DIM,))

# ------------------------------------------------------------
# Single updates
# ------------------------------------------------------------

function ppo_setup(h, B, rng)
    alg = PPO(; batch_size = B)
    model = ActorCriticModel(OBS_SPACE, ACT_SPACE; hidden_dims = [h, h])
    batch = (
        randn(rng, Float32, OBS_DIM, B), randn(rng, Float32, ACT_DIM, B),
        randn(rng, Float32, B), randn(rng, Float32, B),
        -abs.(randn(rng, Float32, B)) .- 3, randn(rng, Float32, B),
    )
    update = (learner, batch) -> ppo_update(alg, model, learner, batch, AD)
    return alg, model, batch, update
end

function sac_setup(h, B, rng)
    alg = SAC(; batch_size = B)
    model = SACModel(OBS_SPACE, ACT_SPACE; hidden_dims = [h, h])
    batch = (
        observations = randn(rng, Float32, OBS_DIM, B),
        actions = clamp.(randn(rng, Float32, ACT_DIM, B), -1, 1),
        rewards = randn(rng, Float32, B),
        terminated = collect(rand(rng, B) .< 0.05),
        truncated = zeros(Bool, B),
        next_observations = randn(rng, Float32, OBS_DIM, B),
    )
    target_entropy = Drill.get_target_entropy(alg.ent_coef, ACT_SPACE)
    update = (learner, batch) -> sac_update(alg, model, learner, batch, AD, target_entropy, Val(true))
    return alg, model, batch, update
end

function timed_step(step, learner, batch)
    t = @elapsed begin
        learner, metrics = step(learner, batch)
        Drill.host_metrics(metrics)
    end
    return learner, t
end

function bench_update(alg_name, h, B)
    rng = Xoshiro(0)
    alg, model, batch, update = alg_name == "ppo" ? ppo_setup(h, B, rng) : sac_setup(h, B, rng)
    ps, st = Lux.setup(rng, model)
    learner = Drill.init_learner(
        alg, model, DEV(ps), DEV(st);
        rng = Drill.device_rng(DEV, Xoshiro(1)), device = DEV,
    )
    batch = DEV(batch)
    step, t_first = if DEV isa Lux.ReactantDevice
        t = @elapsed compiled = Reactant.@compile update(learner, batch)
        compiled, t
    else
        update, @elapsed(update(deepcopy(learner), batch))
    end
    for _ in 1:5
        learner, _ = timed_step(step, learner, batch)
    end
    learner, t_one = timed_step(step, learner, batch)
    n = clamp(round(Int, 2.0 / t_one), 10, 2000)
    times = Float64[]
    GC.gc()
    for _ in 1:n
        learner, t = timed_step(step, learner, batch)
        push!(times, t)
    end
    @printf(
        "%s %s h=%d B=%d params=%d first=%.2fs median=%.1fus min=%.1fus n=%d\n",
        BACKEND, alg_name, h, B, Lux.parameterlength(model), t_first,
        1.0e6 * median(times), 1.0e6 * minimum(times), n,
    )
    flush(stdout)
    return times
end

# ------------------------------------------------------------
# Full PPO iterations
# ------------------------------------------------------------

mutable struct RandomEnv <: AbstractEnv
    n::Int
    done::Bool
    rng::Xoshiro
end

DrillInterface.observation_space(::RandomEnv) = OBS_SPACE
DrillInterface.action_space(::RandomEnv) = ACT_SPACE
DrillInterface.terminated(env::RandomEnv) = env.done
DrillInterface.truncated(::RandomEnv) = false
DrillInterface.get_info(::RandomEnv) = Dict{String, Any}()
DrillInterface.observe(env::RandomEnv) = randn(env.rng, Float32, OBS_DIM)

function DrillInterface.reset!(env::RandomEnv; seed = nothing)
    env.n = 0
    env.done = false
    return nothing
end

function DrillInterface.act!(env::RandomEnv, action::AbstractArray)
    env.n += 1
    env.done = env.n >= 200
    return -sum(abs2, action)
end

function bench_iteration(h; n_envs = 8, n_steps = 256, iterations = 6)
    env = BroadcastedParallelEnv([RandomEnv(0, false, Xoshiro(i)) for i in 1:n_envs])
    model = ActorCriticModel(OBS_SPACE, ACT_SPACE; hidden_dims = [h, h])
    alg = PPO(; n_steps, batch_size = 64, epochs = 10)
    device_kw = DEV isa Lux.ReactantDevice ? (; device = DEV) : (;)
    ad_type = BACKEND == "zygote" ? AutoZygote() : AutoEnzyme()
    cache = init(
        RLProblem(env, model), alg;
        max_steps = n_steps * n_envs * iterations, verbosity = 0,
        logger = NoTrainingLogger(), rng = Xoshiro(0), ad_type, device_kw...,
    )
    times = [@elapsed(step!(cache)) for _ in 1:iterations]
    @printf(
        "%s iteration h=%d first=%.1fs median=%.0fms min=%.0fms\n",
        BACKEND, h, times[1], 1.0e3 * median(times[2:end]), 1.0e3 * minimum(times[2:end]),
    )
    return times
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

parse_list(s) = parse.(Int, split(s, ","))

if MODE == "update"
    alg_name = ARGS[3]
    widths = parse_list(get(ARGS, 4, "64,256,1024,2048"))
    batches = parse_list(get(ARGS, 5, "64,1024"))
    for h in widths, B in batches
        bench_update(alg_name, h, B)
    end
elseif MODE == "iteration"
    bench_iteration(parse(Int, ARGS[3]))
else
    error("Unknown mode $MODE; use update or iteration")
end
