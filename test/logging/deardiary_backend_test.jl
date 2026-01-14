@testitem "DearDiary logger converts and logs without error" begin
    using DearDiary
    using DRiL
    mktempdir() do dir
        # Initialize database in temp directory
        db_path = joinpath(dir, "test.db")
        DearDiary.initialize_database(; file_name = db_path)

        # Create project and experiment
        project_id, _ = DearDiary.create_project("DRiL Test Project")
        experiment_id, _ = DearDiary.create_experiment(
            project_id, DearDiary.IN_PROGRESS, "Test Experiment"
        )

        # Convert experiment ID to logger backend
        lg = convert(DRiL.AbstractTrainingLogger, experiment_id)

        # Test basic logging operations
        DRiL.set_step!(lg, 1)
        DRiL.log_scalar!(lg, "loss", 0.5)
        DRiL.log_scalar!(lg, "reward", 10.0)

        # Test dict logging
        DRiL.set_step!(lg, 2)
        DRiL.log_metrics!(lg, Dict("loss" => 0.4, "reward" => 15.0))

        # Test hyperparameter logging
        DRiL.log_hparams!(
            lg, Dict("lr" => 0.01, "gamma" => 0.99, "algorithm" => "PPO"),
            ["env/ep_rew_mean"]
        )

        # Test increment step
        DRiL.increment_step!(lg, 1)
        DRiL.log_scalar!(lg, "loss", 0.3)

        DRiL.flush!(lg)
        DRiL.close!(lg)

        # Verify data was logged
        iterations = DearDiary.get_iterations(experiment_id)
        @test length(iterations) >= 3

        # Check metrics exist in first iteration
        first_iteration = first(iterations)
        metrics = DearDiary.get_metrics(first_iteration.id)
        @test length(metrics) >= 1

        @test isfile(db_path)

        # Clean up database connection
        DearDiary.close_database()
    end
end
