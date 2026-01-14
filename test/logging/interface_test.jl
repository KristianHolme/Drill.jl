@testitem "NoTrainingLogger basics" begin
    lg = NoTrainingLogger()
    set_step!(lg, 10)
    log_scalar!(lg, "a", 1.0)
    log_metrics!(lg, Dict("b" => 2.0))
    log_hparams!(lg, Dict("lr" => 0.01, "gamma" => 0.99), ["metric1"])
    flush!(lg)
    close!(lg)
    @test true
end
