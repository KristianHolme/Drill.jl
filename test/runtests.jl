using TestItemRunner

# Quality assurance tests
@testitem "Code quality (Aqua.jl)" tags = [:quality] begin
    using Aqua, DRiL
    Aqua.test_all(DRiL)
end

@testitem "Code linting (JET.jl)" tags = [:quality] begin
    using JET, DRiL

    """
    Filter out known false positives from JET analysis.

    TimerOutputs.@timeit macro generates code that conditionally defines `accumulated_data`
    guarded by an `enabled` flag, then uses it in a finally block guarded by the same flag.
    JET can't prove the variable is defined in the second branch, so it reports a false
    positive "local variable may be undefined" warning.
    """
    function filter_false_positives(reports)
        return filter(reports) do r
            if r isa JET.UndefVarErrorReport && r.maybeundef
                varname = string(r.var)
                # Filter out accumulated_data from TimerOutputs @timeit macro
                if startswith(varname, "accumulated_data")
                    return false
                end
            end
            return true
        end
    end

    if get(ENV, "JET_STRICT", "") == "1"
        report = JET.report_package(DRiL; target_modules = (DRiL,), toplevel_logger = nothing)
        filtered_reports = filter_false_positives(JET.get_reports(report))
        @test isempty(filtered_reports)
    else
        # Advisory mode: print report but don't fail CI
        report = JET.report_package(DRiL; target_modules = (DRiL,), toplevel_logger = nothing)
        filtered_reports = filter_false_positives(JET.get_reports(report))
        if !isempty(filtered_reports)
            println("JET found $(length(filtered_reports)) potential issues:")
            println(report)
        end
        @test true
    end
end

# Run all tests
# @run_package_tests
# Run only tests with :ad_backends tag
@run_package_tests filter = ti -> :ad_backends ∉ ti.tags
