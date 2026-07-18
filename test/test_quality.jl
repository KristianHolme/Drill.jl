using Test
using Drill
using Aqua
using ExplicitImports
using JET

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(Drill)
end

@testset "ExplicitImports" begin
    modules = Any[
        Drill,
        Drill.Adapters,
        Drill.Models,
        Drill.Buffers,
        Drill.DrillLogging,
        Drill.Algorithms,
        Drill.Solve,
        Drill.Wrappers,
        Drill.Utils,
    ]
    # Aqua / weakdeps may load extensions; check any that are present.
    # Extensions must use explicit imports (see AGENTS.md).
    for ext_name in (
            :Drill_PrettyTablesExt,
            :Drill_WandbExt,
            :Drill_TensorBoardLoggerExt,
            :Drill_DearDiaryExt,
            :Drill_ReactantExt,
        )
        ext = Base.get_extension(Drill, ext_name)
        ext === nothing || push!(modules, ext)
    end
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
