using Drill
using Test
using ParallelTestRunner

"""
If `DRILL_EXCLUDE_TEST_LOGGING=true` and no explicit test filters were passed,
run every discovered test except `test_logging`. Used by CI so logging can run
in a separate serial step with live (unbuffered) output.
"""
function _args_for_ci(args::Vector{String})
    if get(ENV, "DRILL_EXCLUDE_TEST_LOGGING", "") != "true"
        return args
    end
    positional = filter(a -> !startswith(a, "-"), args)
    if !isempty(positional)
        return args
    end
    testdir = @__DIR__
    names = String[]
    for (root, _, files) in walkdir(testdir)
        for file in files
            endswith(file, ".jl") || continue
            file == "runtests.jl" && continue
            rel = relpath(joinpath(root, file), testdir)
            name = replace(replace(rel, r"\.jl$" => ""), '\\' => '/')
            name == "test_logging" && continue
            push!(names, name)
        end
    end
    sort!(names)
    return vcat(args, names)
end

ParallelTestRunner.runtests(Drill, _args_for_ci(copy(ARGS)))
