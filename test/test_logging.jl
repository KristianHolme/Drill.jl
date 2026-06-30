using Test
using Drill

@testset "NoTrainingLogger basics" begin
    lg = NoTrainingLogger()
    set_step!(lg, 10)
    log_scalar!(lg, "a", 1.0)
    log_metrics!(lg, Dict("b" => 2.0))
    log_hparams!(lg, Dict("lr" => 0.01, "gamma" => 0.99), ["metric1"])
    flush!(lg)
    close!(lg)
    @test true
end

@testset "Wandb logger converts and logs offline without error" begin
    using Wandb
    ENV["WANDB_MODE"] = "offline"
    ENV["WANDB_SILENT"] = "true"
    haskey(ENV, "WANDB_API_KEY") || (ENV["WANDB_API_KEY"] = "DUMMY")
    mktempdir() do dir
        ENV["WANDB_DIR"] = dir
        raw = WandbLogger(; project = "dril_test", name = "unit", mode = "offline", dir = dir)
        lg = convert(Drill.AbstractTrainingLogger, raw)
        Drill.set_step!(lg, 1)
        Drill.log_scalar!(lg, "y", 2.0)
        Drill.log_hparams!(lg, Dict("lr" => 0.01, "gamma" => 0.99), ["env/ep_rew_mean"])
        Drill.flush!(lg)
        Drill.close!(lg)
        @test isdir(dir)
    end
end

@testset "TB logger converts and logs without error" begin
    using TensorBoardLogger
    mktempdir() do dir
        raw = TBLogger(dir, tb_increment)
        lg = convert(Drill.AbstractTrainingLogger, raw)
        Drill.set_step!(lg, 1)
        Drill.log_scalar!(lg, "x", 1.0)
        Drill.log_hparams!(lg, Dict("lr" => 0.01, "gamma" => 0.99), ["env/ep_rew_mean"])
        Drill.flush!(lg)
        Drill.close!(lg)
        @test isdir(dir)
    end
end

@testset "Training progress options" begin
    @testset "plain_progress_options disables colors" begin
        opts = plain_progress_options()
        @test opts.color == :normal
        @test opts.valuecolor == :normal
        @test opts.output === stderr
    end

    @testset "make_training_progress_meter respects options" begin
        io = IOBuffer()
        opts = plain_progress_options(output = io)
        meter = make_training_progress_meter(10, 1, opts)
        @test meter.enabled
        @test meter.output === io
        @test meter.color == :normal
        @test meter.showspeed
    end

    @testset "make_training_progress_meter disabled when verbose is 0" begin
        meter = make_training_progress_meter(10, 0)
        @test !meter.enabled
    end

    @testset "progress_next! avoids ANSI color codes" begin
        io = IOBuffer()
        opts = plain_progress_options(output = io)
        meter = make_training_progress_meter(4, 2, opts)
        progress_next!(meter, opts, step = 2, showvalues = [("loss", 0.5)])
        output = String(take!(io))
        @test occursin("Training", output)
        @test !occursin(r"\e\[[0-9;]*m", output)
    end
end

@testset "DearDiary logger converts and logs without error" begin
    using DearDiary
    mktempdir() do dir
        db_path = joinpath(dir, "test.db")
        DearDiary.initialize_database(; file_name = db_path)

        project_id, _ = DearDiary.create_project("Drill Test Project")
        experiment_id, _ = DearDiary.create_experiment(
            project_id, DearDiary.IN_PROGRESS, "Test Experiment"
        )

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
        Drill.close!(lg)

        iterations = DearDiary.get_iterations(experiment_id)
        @test length(iterations) >= 3

        first_iteration = first(iterations)
        metrics = DearDiary.get_metrics(first_iteration.id)
        @test length(metrics) >= 1

        @test isfile(db_path)

        DearDiary.close_database()
    end
end
