using TestItemRunner

# Use a shared CondaPkg environment for Wandb tests unless explicitly overridden.
if !haskey(ENV, "JULIA_CONDAPKG_ENV")
    ENV["JULIA_CONDAPKG_ENV"] = "@drill-wandb-tests"
end

const DRILL_TEST_GROUP = lowercase(get(ENV, "DRILL_TEST_GROUP", "all"))

function include_test_item(ti)
    if :ad_backends in ti.tags
        return false
    elseif DRILL_TEST_GROUP == "all"
        return true
    elseif DRILL_TEST_GROUP == "core"
        return :quality ∉ ti.tags
    elseif DRILL_TEST_GROUP == "quality"
        return :quality in ti.tags
    elseif DRILL_TEST_GROUP == "fast"
        return :quality ∉ ti.tags && :wandb ∉ ti.tags
    elseif DRILL_TEST_GROUP == "wandb"
        return :wandb in ti.tags
    end

    throw(ArgumentError("Unknown DRILL_TEST_GROUP=\"$DRILL_TEST_GROUP\". Supported values: all, core, quality, fast, wandb"))
end

# Quality assurance tests
@testitem "Code quality (Aqua.jl)" tags = [:quality] begin
    using Aqua, Drill
    Aqua.test_all(Drill)
end

@testitem "Code linting (JET.jl)" tags = [:quality] begin
    using JET, Drill

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
        report = JET.report_package(Drill; target_modules = (Drill,), toplevel_logger = nothing)
        filtered_reports = filter_false_positives(JET.get_reports(report))
        @test isempty(filtered_reports)
    else
        # Advisory mode: print report but don't fail CI
        report = JET.report_package(Drill; target_modules = (Drill,), toplevel_logger = nothing)
        filtered_reports = filter_false_positives(JET.get_reports(report))
        if !isempty(filtered_reports)
            println("JET found $(length(filtered_reports)) potential issues:")
            println(report)
        end
        @test true
    end
end

# Run selected test group.
@run_package_tests filter = include_test_item
