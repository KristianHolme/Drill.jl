# Verbosity showcase: short PPO runs on CartPole with different console settings.
#
# From the repo root:
#   julia --project=examples examples/verbosity.jl
#
# Or from this directory:
#   julia --project=. verbosity.jl

using ClassicControlEnvironments
using Drill
using PrettyTables  # enables Drill_PrettyTablesExt for verbosity.table = true
using Random
using Zygote

const N_ENVS = 4
const N_STEPS = 64
const EPOCHS = 2
const BATCH_SIZE = 64
const UPDATES = 3
const TOTAL_STEPS = N_STEPS * N_ENVS * UPDATES
const SEED = 42

function make_problem(seed::Int)
    rng = Random.Xoshiro(seed)
    envs = [CartPoleEnv(; rng = Random.Xoshiro(seed + i)) for i in 1:N_ENVS]
    env = MonitorWrapperEnv(BroadcastedParallelEnv(envs))
    reset!(env)
    model = ActorCriticModel(observation_space(env), action_space(env); hidden_dims = [64, 64])
    alg = PPO(;
        n_steps = N_STEPS,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        learning_rate = 3.0f-4,
    )
    return RLProblem(env, model), alg, rng
end

function run_demo(label::AbstractString, verbosity)
    println()
    println("="^72)
    println(label)
    println("verbosity = ", verbosity)
    println("="^72)
    prob, alg, rng = make_problem(SEED)
    cache = init(prob, alg; max_steps = TOTAL_STEPS, verbosity = verbosity, rng = rng)
    solve!(cache)
    println("finished: steps_taken=$(cache.steps_taken), retcode=$(cache.retcode)")
    return cache
end

function main()
    println("Drill verbosity examples (CartPole PPO, $(TOTAL_STEPS) steps)")
    println("Defaults merge with (; meter = 2, table = false, timer = 0)")

    # Int shorthand: meter only; table false / timer 0
    run_demo("1) Quiet — Int shorthand verbosity = 0", 0)

    run_demo(
        "2) Simple progress bar — meter = 1",
        (; meter = 1, table = false, timer = 0),
    )

    run_demo(
        "3) Progress bar + live stats (showvalues) — meter = 2",
        (; meter = 2, table = false, timer = 0),
    )

    run_demo(
        "4) PrettyTables dump each update — table = true",
        (; meter = 0, table = true, timer = 0),
    )

    run_demo(
        "5) TimerOutputs at end — timer = 2",
        (; meter = 0, table = false, timer = 2),
    )

    run_demo(
        "6) Combined rich console — meter = 2, table = true, timer = 2",
        (; meter = 2, table = true, timer = 2),
    )

    println()
    println("Done. Partial NamedTuples merge with defaults, e.g.:")
    println("  solve(prob, alg; max_steps, verbosity = (; meter = 1))")
    println("  # => Verbosity(meter=1, table=false, timer=0)")
    return nothing
end

main()
