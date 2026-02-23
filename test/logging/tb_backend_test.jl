@testitem "TB logger converts and logs without error" begin
    using TensorBoardLogger
    using Drill
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
