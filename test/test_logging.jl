using Test
using Drill

# Temporary diagnostics: flushed markers so a serial CI step shows live progress.
# ParallelTestRunner buffers worker stdio until the file finishes, so markers only
# help when this file is run directly (see CI "Run test_logging" step).
function _test_logging_diag(msg::AbstractString)
    println(stderr, "[test_logging $(round(time(); digits = 1))] $msg")
    flush(stderr)
    return nothing
end

_test_logging_diag("file start")

@testset "NoTrainingLogger basics" begin
    _test_logging_diag("NoTrainingLogger: begin")
    lg = NoTrainingLogger()
    set_step!(lg, 10)
    log_scalar!(lg, "a", 1.0)
    log_metrics!(lg, Dict("b" => 2.0))
    log_hparams!(lg, Dict("lr" => 0.01, "gamma" => 0.99), ["metric1"])
    flush!(lg)
    close!(lg)
    @test true
    _test_logging_diag("NoTrainingLogger: done")
end

@testset "Wandb logger converts and logs offline without error" begin
    _test_logging_diag("Wandb: using Wandb")
    using Wandb
    _test_logging_diag("Wandb: package loaded")
    ENV["WANDB_MODE"] = "offline"
    ENV["WANDB_SILENT"] = "true"
    haskey(ENV, "WANDB_API_KEY") || (ENV["WANDB_API_KEY"] = "DUMMY")
    mktempdir() do dir
        ENV["WANDB_DIR"] = dir
        _test_logging_diag("Wandb: creating WandbLogger")
        raw = WandbLogger(; project = "dril_test", name = "unit", mode = "offline", dir = dir)
        _test_logging_diag("Wandb: WandbLogger created")
        lg = convert(Drill.AbstractTrainingLogger, raw)
        Drill.set_step!(lg, 1)
        Drill.log_scalar!(lg, "y", 2.0)
        Drill.log_hparams!(lg, Dict("lr" => 0.01, "gamma" => 0.99), ["env/ep_rew_mean"])
        Drill.flush!(lg)
        _test_logging_diag("Wandb: before close!")
        Drill.close!(lg)
        _test_logging_diag("Wandb: after close!")
        @test isdir(dir)
    end
    _test_logging_diag("Wandb: testset done")
end

@testset "TB logger converts and logs without error" begin
    _test_logging_diag("TB: using TensorBoardLogger")
    using TensorBoardLogger
    _test_logging_diag("TB: package loaded")
    mktempdir() do dir
        _test_logging_diag("TB: creating TBLogger")
        raw = TBLogger(dir, tb_increment)
        _test_logging_diag("TB: TBLogger created")
        lg = convert(Drill.AbstractTrainingLogger, raw)
        Drill.set_step!(lg, 1)
        Drill.log_scalar!(lg, "x", 1.0)
        Drill.log_hparams!(lg, Dict("lr" => 0.01, "gamma" => 0.99), ["env/ep_rew_mean"])
        Drill.flush!(lg)
        _test_logging_diag("TB: before close!")
        Drill.close!(lg)
        _test_logging_diag("TB: after close!")
        @test isdir(dir)
    end
    _test_logging_diag("TB: testset done")
end

@testset "DearDiary logger converts and logs without error" begin
    _test_logging_diag("DearDiary: using DearDiary")
    using DearDiary
    _test_logging_diag("DearDiary: package loaded")
    mktempdir() do dir
        db_path = joinpath(dir, "test.db")
        _test_logging_diag("DearDiary: initialize_database")
        DearDiary.initialize_database(; file_name = db_path)
        _test_logging_diag("DearDiary: database initialized")

        project_id, _ = DearDiary.create_project("Drill Test Project")
        experiment_id, _ = DearDiary.create_experiment(
            project_id, DearDiary.IN_PROGRESS, "Test Experiment"
        )
        _test_logging_diag("DearDiary: project/experiment created")

        lg = convert(Drill.AbstractTrainingLogger, experiment_id)

        Drill.set_step!(lg, 1)
        Drill.log_scalar!(lg, "loss", 0.5)
        Drill.log_scalar!(lg, "reward", 10.0)

        Drill.set_step!(lg, 2)
        Drill.log_metrics!(lg, Dict("loss" => 0.4, "reward" => 15.0))

        Drill.log_hparams!(
            lg, Dict("lr" => 0.01, "gamma" => 0.99, "algorithm" => "PPO"),
            ["env/ep_rew_mean"]
        )

        Drill.increment_step!(lg, 1)
        Drill.log_scalar!(lg, "loss", 0.3)

        Drill.flush!(lg)
        _test_logging_diag("DearDiary: before close!")
        Drill.close!(lg)
        _test_logging_diag("DearDiary: after close!")

        iterations = DearDiary.get_iterations(experiment_id)
        @test length(iterations) >= 3

        first_iteration = first(iterations)
        metrics = DearDiary.get_metrics(first_iteration.id)
        @test length(metrics) >= 1

        @test isfile(db_path)

        _test_logging_diag("DearDiary: before close_database")
        DearDiary.close_database()
        _test_logging_diag("DearDiary: after close_database")
    end
    _test_logging_diag("DearDiary: testset done")
end

_test_logging_diag("file done")
