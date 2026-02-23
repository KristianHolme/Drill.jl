@testitem "Wandb logger converts and logs offline without error" begin
    using Wandb
    # Force offline; silence auth; provide dummy key for non-interactive runs
    ENV["WANDB_MODE"] = "offline"
    ENV["WANDB_SILENT"] = "true"
    haskey(ENV, "WANDB_API_KEY") || (ENV["WANDB_API_KEY"] = "DUMMY")
    mktempdir() do dir
        # ensure all wandb files are placed in a temp directory
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
