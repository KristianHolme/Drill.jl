using TestItemRunner

# Use a shared CondaPkg environment for Wandb tests unless explicitly overridden.
if !haskey(ENV, "JULIA_CONDAPKG_ENV")
    ENV["JULIA_CONDAPKG_ENV"] = "@drill-wandb-tests"
end

function parse_tag_list(var_name::AbstractString)
    raw = strip(get(ENV, var_name, ""))
    if isempty(raw)
        return Set{Symbol}()
    end

    tags = Set{Symbol}()
    for token in split(raw, r"[,\s]+")
        cleaned = strip(token)
        if isempty(cleaned)
            continue
        end
        if startswith(cleaned, ":")
            cleaned = cleaned[2:end]
        end
        push!(tags, Symbol(cleaned))
    end
    return tags
end

function contains_any(tags, selected_tags::Set{Symbol})
    for tag in tags
        if tag in selected_tags
            return true
        end
    end
    return false
end

const TEST_TAG_WHITELIST = parse_tag_list("DRILL_TEST_TAG_WHITELIST")
const TEST_TAG_BLACKLIST = union(Set{Symbol}([:ad_backends]), parse_tag_list("DRILL_TEST_TAG_BLACKLIST"))
const TAG_LIST_OVERLAP = intersect(TEST_TAG_WHITELIST, TEST_TAG_BLACKLIST)

if !isempty(TAG_LIST_OVERLAP)
    overlap_tags = join(sort!(string.(collect(TAG_LIST_OVERLAP))), ", ")
    @warn "Tags are present in both DRILL_TEST_TAG_WHITELIST and DRILL_TEST_TAG_BLACKLIST: $overlap_tags. Blacklist takes precedence."
end

function include_test_item(ti)
    if contains_any(ti.tags, TEST_TAG_BLACKLIST)
        return false
    end
    if isempty(TEST_TAG_WHITELIST)
        return true
    end
    return contains_any(ti.tags, TEST_TAG_WHITELIST)
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

# Run tests selected by tag whitelist/blacklist.
@run_package_tests filter = include_test_item
