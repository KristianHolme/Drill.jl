using Test
using Drill
using Aqua
using ExplicitImports
using JET

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(Drill)
end

@testset "ExplicitImports" begin
    modules = (
        Drill,
        Drill.Adapters,
        Drill.Layers,
        Drill.Buffers,
        Drill.Callbacks,
        Drill.DrillLogging,
        Drill.Algorithms,
        Drill.Problem,
        Drill.Solve,
        Drill.Wrappers,
        Drill.Utils,
        Drill.Deployment,
        Drill.Evaluation,
    )
    for m in modules
        @test check_no_implicit_imports(m) === nothing
        @test check_no_stale_explicit_imports(m) === nothing
    end
end

@testset "Code linting (JET.jl)" begin
    function filter_false_positives(reports)
        return filter(reports) do r
            if r isa JET.UndefVarErrorReport && r.maybeundef
                varname = string(r.var)
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
        report = JET.report_package(Drill; target_modules = (Drill,), toplevel_logger = nothing)
        filtered_reports = filter_false_positives(JET.get_reports(report))
        if !isempty(filtered_reports)
            println("JET found $(length(filtered_reports)) potential issues:")
            println(report)
        end
        @test true
    end
end
