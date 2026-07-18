using Drill
using Test
using ParallelTestRunner

"""
Skip `test_logging` on GitHub Actions. WandbLogger leaves a hung `wandb-core`
process that blocks ParallelTestRunner until the job timeout. See the tracking
GitHub issue; tests still run locally.
"""
function _args_excluding_logging_on_gha(args::Vector{String})
    if get(ENV, "GITHUB_ACTIONS", "") != "true"
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

ParallelTestRunner.runtests(Drill, _args_excluding_logging_on_gha(copy(ARGS)))
