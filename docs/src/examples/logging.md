# Logging & Experiment Tracking

Drill.jl supports three logging backends for tracking your experiments:

- **TensorBoard** - Local logging with web-based visualization
- **Weights & Biases (W&B)** - Cloud-based experiment tracking with team collaboration features
- **DearDiary** - Pure Julia experiment tracking with SQLite backend

All backends automatically log training metrics during `train!()` calls, including:

- Episode rewards and lengths
- Policy/value/entropy losses
- Learning rate, gradient norms
- Frames per second (FPS)

## TensorBoard Example

TensorBoard is great for local development and quick experiments. Logs are stored locally and viewed through a web interface.

### Setup

```julia
using Pkg
Pkg.add("TensorBoardLogger")
```

### Example

```julia
using Drill
using ClassicControlEnvironments
using TensorBoardLogger
using Logging
using Random
using Zygote

# Configuration
LOG_DIR = joinpath(pwd(), "data", "tensorboard")
N_ENVS = 4
TOTAL_TIMESTEPS = 50_000
SEED = 42

# Hyperparameter configs to compare
configs = [
    (name = "default", learning_rate = 3.0f-4, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.0f0),
    (name = "high_lr", learning_rate = 1.0f-3, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.0f0),
    (name = "with_entropy", learning_rate = 3.0f-4, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.01f0),
]

# Run experiments
for config in configs
    # Create parallel environments
    rng = Random.Xoshiro(SEED)
    envs = [CartPoleEnv(; rng = Random.Xoshiro(SEED + i)) for i in 1:N_ENVS]
    env = MonitorWrapperEnv(BroadcastedParallelEnv(envs))
    DrillInterface.reset!(env)

    # Create PPO algorithm
    alg = PPO(;
        learning_rate = config.learning_rate,
        n_steps = config.n_steps,
        batch_size = config.batch_size,
        epochs = config.epochs,
        ent_coef = config.ent_coef,
    )

    # Create TensorBoard logger - each run gets its own subdirectory
    run_dir = joinpath(LOG_DIR, "CartPole_$(config.name)")
    tb_logger = TBLogger(run_dir; min_level = Logging.Info)

    # Create agent with logger
    layer = ActorCriticLayer(observation_space(env), action_space(env))
    agent = Agent(layer, alg; logger = tb_logger, verbose = 1, rng = rng)

    # Log hyperparameters (viewable in TensorBoard HPARAMS tab)
    log_hparams!(
        agent.logger,
        Dict(String(k) => v for (k, v) in pairs(config)),
        ["env/ep_rew_mean", "train/loss"],
    )

    # Train - metrics auto-logged to TensorBoard
    @info "Training" config = config.name log_dir = run_dir
    train!(agent, env, alg, TOTAL_TIMESTEPS)

    # Close logger
    close!(agent.logger)
end
```

### Viewing Results

Run in terminal:

```bash
tensorboard --logdir=data/tensorboard
```

Then open your browser at url printed in the terminal (usually something like `localhost:6006`)

- **SCALARS** tab: Training metrics over time
- **HPARAMS** tab: Compare hyperparameter configurations

---

## Weights & Biases Example

W&B is ideal for cloud-based tracking, team collaboration, and hyperparameter sweeps.

### Setup

1. Create account at [wandb.ai](https://wandb.ai/site)
2. Install and authenticate:

```julia
using Pkg
Pkg.add("Wandb")

using Wandb
Wandb.login()  # Enter API key from https://wandb.ai/authorize
```

### Example

```julia
using Drill
using ClassicControlEnvironments
using Wandb
using Random
using Zygote

# Configuration
WANDB_PROJECT = "Drill-Examples"
N_ENVS = 4
TOTAL_TIMESTEPS = 50_000
SEED = 42

# Hyperparameter configs to compare
configs = [
    (name = "default", learning_rate = 3.0f-4, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.0f0),
    (name = "high_lr", learning_rate = 1.0f-3, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.0f0),
    (name = "with_entropy", learning_rate = 3.0f-4, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.01f0),
]

# Run experiments
for config in configs
    # Create parallel environments
    rng = Random.Xoshiro(SEED)
    envs = [CartPoleEnv(; rng = Random.Xoshiro(SEED + i)) for i in 1:N_ENVS]
    env = MonitorWrapperEnv(BroadcastedParallelEnv(envs))
    reset!(env)

    # Create PPO algorithm
    alg = PPO(;
        learning_rate = config.learning_rate,
        n_steps = config.n_steps,
        batch_size = config.batch_size,
        epochs = config.epochs,
        ent_coef = config.ent_coef,
    )

    # Create W&B logger - each run appears as separate experiment in dashboard
    wb_logger = WandbLogger(;
        project = WANDB_PROJECT,
        name = "CartPole_$(config.name)",
        config = Dict(String(k) => v for (k, v) in pairs(config)),
    )

    # Create agent with logger
    layer = ActorCriticLayer(observation_space(env), action_space(env))
    agent = Agent(layer, alg; logger = wb_logger, verbose = 1, rng = rng)

    # Train - metrics auto-logged to W&B
    @info "Training" config = config.name
    train!(agent, env, alg, TOTAL_TIMESTEPS)

    # Always close logger to finalize the W&B run
    close!(agent.logger)
end
```

### Viewing Results

After training, view results at `https://wandb.ai/<username>/Drill-Examples`

- Compare runs side-by-side
- Use parallel coordinates for hyperparameter analysis
- Share results with team members

---

## DearDiary Example

DearDiary.jl is a pure Julia experiment tracking solution with a portable SQLite backend. It's ideal when you want:

- No external Python dependencies
- Self-hosted, local-first experiment tracking
- Portable database files that can be easily shared or archived

### Setup

```julia
using Pkg
Pkg.add("DearDiary")
```

### Example

```julia
using Drill
using ClassicControlEnvironments
using DearDiary
using Random
using Zygote

# Configuration
DB_PATH = joinpath(pwd(), "data", "experiments.db")
N_ENVS = 4
TOTAL_TIMESTEPS = 50_000
SEED = 42

# Initialize the database (creates file if it doesn't exist)
DearDiary.initialize_database(; file_name = DB_PATH)

# Create a project for your experiments
project_id, _ = DearDiary.create_project("Drill Experiments")

# Hyperparameter configs to compare
configs = [
    (name = "default", learning_rate = 3.0f-4, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.0f0),
    (name = "high_lr", learning_rate = 1.0f-3, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.0f0),
    (name = "with_entropy", learning_rate = 3.0f-4, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.01f0),
]

# Run experiments
for config in configs
    # Create parallel environments
    rng = Random.Xoshiro(SEED)
    envs = [CartPoleEnv(; rng = Random.Xoshiro(SEED + i)) for i in 1:N_ENVS]
    env = MonitorWrapperEnv(BroadcastedParallelEnv(envs))
    DrillInterface.reset!(env)

    # Create PPO algorithm
    alg = PPO(;
        learning_rate = config.learning_rate,
        n_steps = config.n_steps,
        batch_size = config.batch_size,
        epochs = config.epochs,
        ent_coef = config.ent_coef,
    )

    # Create DearDiary experiment - each run is a separate experiment
    experiment_id, _ = DearDiary.create_experiment(
        project_id,
        DearDiary.IN_PROGRESS,
        "CartPole_$(config.name)"
    )

    # Create agent with the experiment ID as logger
    layer = ActorCriticLayer(observation_space(env), action_space(env))
    agent = Agent(layer, alg; logger = experiment_id, verbose = 1, rng = rng)

    # Log hyperparameters
    log_hparams!(
        agent.logger,
        Dict(String(k) => v for (k, v) in pairs(config)),
        ["env/ep_rew_mean", "train/loss"],
    )

    # Train - metrics auto-logged to DearDiary
    @info "Training" config = config.name experiment_id = experiment_id
    train!(agent, env, alg, TOTAL_TIMESTEPS)

    # Mark experiment as complete and close logger
    DearDiary.update_experiment(experiment_id, DearDiary.FINISHED, nothing, nothing, nothing)
    close!(agent.logger)
end

# Don't forget to close the database when done
DearDiary.close_database()
```

### Viewing Results

DearDiary stores data in a SQLite database that you can query directly:

```julia
using DearDiary

# Open existing database
DearDiary.initialize_database(; file_name = "data/experiments.db")

# List all experiments in a project
experiments = DearDiary.get_experiments(project_id)

# Get iterations (training steps) for an experiment
iterations = DearDiary.get_iterations(experiment_id)

# Get metrics for a specific iteration
metrics = DearDiary.get_metrics(iteration.id)
for m in metrics
    println("$(m.key): $(m.value)")
end

# Get hyperparameters
params = DearDiary.get_parameters(iteration.id)
```

You can also use any SQLite browser to explore the database file directly, or start the built-in REST API server:

```julia
DearDiary.run()  # Starts server on localhost:9000
```

---

## MultiProgressManagers (training progress)

Experiment loggers (TensorBoard, W&B, DearDiary) record metrics after each policy update. [MultiProgressManagers.jl](https://github.com/KristianHolme/MultiProgressManagers.jl) complements them by tracking **environment steps** during rollouts: progress is written to a SQLite database and can be viewed in a Tachikoma terminal dashboard. This can be useful when you are doing many training runs in parallel and want to get an overview of the total progress and progress of individual runs. See the package readme for installation instructions.

Drill integrates via a package extension: when both `MultiProgressManagers` and `Drill` are loaded, Julia loads `MultiProgressManagersDrillExt`. The helper `create_dril_callback` lives in that extension, so take it with `Base.get_extension` (see the example below). The callback hooks into `on_step` during trajectory collection so the dashboard advances with each parallel-env step. Choose `get_task(manager, i, :local)` for same-process training (shown below) or `:remote` for `Distributed` workers (see the MultiProgressManagers README).

### Setup

The package may be installed from the registry when available, or directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/KristianHolme/MultiProgressManagers.jl")
```

### Example (hyperparameter sweep)

The loop below trains several PPO configurations in sequence; each configuration gets its own task in one `ProgressManager`, so the dashboard shows one bar per run.

```julia
using Drill
using ClassicControlEnvironments
using Random
using Zygote
using MultiProgressManagers

N_ENVS = 4
N_STEPS = 64   # rollout length per PPO iteration
TOTAL_TIMESTEPS = N_STEPS * N_ENVS * 5  # five PPO updates; divisible by N_ENVS
SEED = 42

configs = [
    (name = "default", learning_rate = 3.0f-4, ent_coef = 0.0f0),
    (name = "high_lr", learning_rate = 1.0f-3, ent_coef = 0.0f0),
    (name = "with_entropy", learning_rate = 3.0f-4, ent_coef = 0.01f0),
]

mpm_drill = Base.get_extension(MultiProgressManagers, :MultiProgressManagersDrillExt)
mpm_drill === nothing && error("MultiProgressManagersDrillExt not loaded; use Drill and MultiProgressManagers in the same process")

mktempdir() do dir
    db_path = joinpath(dir, "drill_cartpole_sweep.db")
    manager = ProgressManager(
        "CartPole PPO sweep",
        length(configs);
        db_path = db_path,
        description = "Hyperparameter sweep (Drill + MultiProgressManagers)",
        task_descriptions = [string(c.name) for c in configs],
    )
    for (i, config) in enumerate(configs)
        rng = Random.Xoshiro(SEED + i)
        envs = [CartPoleEnv(; rng = Random.Xoshiro(SEED + 100 * i + j)) for j in 1:N_ENVS]
        env = MonitorWrapperEnv(BroadcastedParallelEnv(envs))
        DrillInterface.reset!(env)

        alg = PPO(;
            n_steps = N_STEPS,
            batch_size = 32,
            epochs = 2,
            learning_rate = config.learning_rate,
            ent_coef = config.ent_coef,
        )
        layer = ActorCriticLayer(observation_space(env), action_space(env))

        task = get_task(manager, i, :local)
        progress_cb = mpm_drill.create_dril_callback(task)
        agent = Agent(layer, alg; verbose = 0, rng = rng)
        train!(agent, env, alg, TOTAL_TIMESTEPS; callbacks = [progress_cb])
    end
    finish!(manager)
end
```

For how to open the Tachikoma dashboard or the `mpm` CLI on the database file, see **Viewing the Dashboard** in the [MultiProgressManagers.jl README](https://github.com/KristianHolme/MultiProgressManagers.jl).

---

## Custom Logging

You can log additional metrics using the logging interface. The logger is accessible via `agent.logger`.

### Available Functions

```julia
# Log a single scalar value
log_scalar!(agent.logger, "custom/my_metric", value)

# Log multiple values at once
log_metrics!(agent.logger, Dict(
    "custom/metric1" => value1,
    "custom/metric2" => value2,
))

# Log hyperparameters (typically done once at start)
log_hparams!(agent.logger, hparams_dict, metric_names)

# Manually set or increment the step counter
set_step!(agent.logger, step)
increment_step!(agent.logger, delta)
```

### Example: Logging Custom Metrics in a Callback

```julia
using Drill

# Define a custom callback that logs additional metrics
struct CustomMetricsCallback <: AbstractCallback end

function Drill.on_rollout_end(cb::CustomMetricsCallback, locals::Dict)
    agent = locals["agent"]
    env = locals["env"]

    # Log custom metrics
    log_scalar!(agent.logger, "custom/episode_count", length(env.episode_stats.episode_returns))

    # Log environment-specific info
    if !isempty(env.episode_stats.episode_returns)
        log_scalar!(agent.logger, "custom/best_return", maximum(env.episode_stats.episode_returns))
    end

    return true  # Continue training
end

# Use the callback during training
callbacks = [CustomMetricsCallback()]
train!(agent, env, alg, total_steps; callbacks)
```

### No Logger (Default)

By default, agents use `NoTrainingLogger()` which discards all log calls. You don't need to pass anything to disable logging:

```julia
# Logging is disabled by default
agent = Agent(layer, alg)

# Equivalent to:
agent = Agent(layer, alg; logger = NoTrainingLogger())
```

### Choosing a Backend

| Feature        | TensorBoard          | W&B                 | DearDiary     |
| -------------- | -------------------- | ------------------- | ------------- |
| Dependencies   | Python (tensorboard) | Python (wandb)      | Pure Julia    |
| Storage        | Local files          | Cloud               | Local SQLite  |
| Real-time view | Yes (web UI)         | Yes (web dashboard) | Query API     |
| Collaboration  | Manual share         | Built-in            | Share DB file |
| Cost           | Free                 | Free tier + paid    | Free          |
