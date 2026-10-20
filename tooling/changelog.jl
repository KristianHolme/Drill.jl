# Adapted from Makie.jl's `tooling/changelog.jl` and `changelog/README.md`
# (https://github.com/MakieOrg/Makie.jl, MIT License,
# Copyright (c) 2018-2021: Simon Danisch, Julius Krumbiegel).
# Changes: per-package entry dirs (Drill, DrillInterface), `release --dry-run`,
# and a `check-pr` command for CI.

module Changelog

    using Dates

    const ROOT = normpath(joinpath(@__DIR__, ".."))
    const PR_URL = "https://github.com/KristianHolme/Drill.jl/pull"

    # Each registered package has its own entry dir and CHANGELOG.md.
    const PACKAGES = Dict("Drill" => ROOT, "DrillInterface" => joinpath(ROOT, "DrillInterface"))
    package_root(pkg) = haskey(PACKAGES, pkg) ? PACKAGES[pkg] : error("Unknown package `$pkg`, expected one of $(join(sort!(collect(keys(PACKAGES))), ", ")).")
    entry_dir(pkg) = joinpath(package_root(pkg), "changelog")
    changelog_path(pkg) = joinpath(package_root(pkg), "CHANGELOG.md")
    relpath_entry(pkg, file) = pkg == "Drill" ? "changelog/$file" : "$pkg/changelog/$file"
    const ENTRY_DIR = entry_dir("Drill")
    const CHANGELOG_PATH = changelog_path("Drill")
    const NON_ENTRY_FILES = ["README.md"]

    const SECTIONS = ["Breaking", "Added", "Changed", "Deprecated", "Removed", "Fixed"]

    struct Entry
        pr::Int
        section::Union{Nothing, String}
        lines::Vector{String}
    end

    function entry_files(dir = ENTRY_DIR)
        files = filter(∉(NON_ENTRY_FILES), readdir(dir))
        for file in files
            occursin(r"^[1-9]\d*\.md$", file) ||
                error("Invalid file `changelog/$file`, entry files must be named `<PR number>.md`.")
        end
        return sort!(joinpath.(dir, files); by = pr_number, rev = true)
    end

    pr_number(path) = parse(Int, chopsuffix(basename(path), ".md"))

    function parse_entry_file(path)
        pr = pr_number(path)
        lines = readlines(path)
        file_error(msg) = error("`changelog/$pr.md`: $msg")

        entries = Entry[]
        section = nothing
        empty_section_line = 0
        continuable = false
        for (i, line) in enumerate(lines)
            heading = match(r"^## (.*)$", line)
            if isempty(strip(line))
                continue
            elseif heading !== nothing
                empty_section_line > 0 && file_error("Section on line $empty_section_line has no entries.")
                section = String(strip(heading[1]))
                section in SECTIONS ||
                    file_error("Invalid section `## $section` on line $i, allowed are $(join(("`## $s`" for s in SECTIONS), ", ")).")
                empty_section_line = i
                continuable = false
            elseif occursin(r"^- \S", line)
                push!(entries, Entry(pr, section, [rstrip(line)]))
                empty_section_line = 0
                continuable = true
            elseif occursin(r"^\s+\S", line) && continuable
                push!(entries[end].lines, rstrip(line))
            else
                file_error(
                    "Invalid line $i: $(repr(line))\n" *
                        "Every line must be a `## <section>` heading, start a `- ` entry, be an indented continuation of the previous entry, or be blank."
                )
            end
        end
        empty_section_line > 0 && file_error("Section on line $empty_section_line has no entries.")
        isempty(entries) && file_error("No entries found, add at least one `- ` entry.")

        occursin(Regex("$(PR_URL)/$pr\\b"), join(lines, "\n")) &&
            file_error("Must not link to PR #$pr, the link is added automatically.")

        return entries
    end

    function render_entry(entry::Entry)
        first_line = "$(entry.lines[1]) [#$(entry.pr)]($PR_URL/$(entry.pr))"
        return [first_line; entry.lines[2:end]]
    end

    render_entries(entries) = mapreduce(render_entry, vcat, entries; init = String[])

    function join_blocks(blocks)
        nonempty = filter(!isempty, blocks)
        return isempty(nonempty) ? String[] : reduce((a, b) -> [a; ""; b], nonempty)
    end

    function unreleased_lines(dir = ENTRY_DIR)
        entries = mapreduce(parse_entry_file, vcat, entry_files(dir); init = Entry[])
        unsectioned = render_entries(filter(e -> e.section === nothing, entries))
        sectioned = map(SECTIONS) do section
            section_entries = filter(e -> e.section == section, entries)
            isempty(section_entries) ? String[] : ["### $section"; ""; render_entries(section_entries)]
        end
        return join_blocks([[unsectioned]; sectioned])
    end

    function insert_section(changelog_lines, section_lines)
        header_idx = findfirst(==("# Changelog"), changelog_lines)
        header_idx === nothing && error("Could not find the `# Changelog` header in CHANGELOG.md.")
        rest = changelog_lines[(header_idx + 1):end]
        first_content = something(findfirst(!isempty ∘ strip, rest), length(rest) + 1)
        return [changelog_lines[1:header_idx]; ""; section_lines; ""; rest[first_content:end]]
    end

    function release!(version::VersionNumber; changelog_path = CHANGELOG_PATH, dir = ENTRY_DIR, date = today())
        entries = unreleased_lines(dir)
        isempty(entries) && @warn "No changelog entries found in $dir."
        section = ["## [$version] - $(Dates.format(date, "yyyy-mm-dd"))"; ""; entries]
        write(changelog_path, join(insert_section(readlines(changelog_path), section), "\n") * "\n")
        rm.(entry_files(dir))
        return
    end

    # Release into a temporary copy and return the new `## [version]` section.
    function preview(pkg, version::VersionNumber)
        return mktempdir() do tmp
            changelog_copy = joinpath(tmp, "CHANGELOG.md")
            dir_copy = joinpath(tmp, "changelog")
            cp(changelog_path(pkg), changelog_copy)
            cp(entry_dir(pkg), dir_copy)
            release!(version; changelog_path = changelog_copy, dir = dir_copy)
            isempty(entry_files(dir_copy)) || error("Entry files were not removed after release.")
            released = readlines(changelog_copy)
            first_section = findfirst(startswith("## "), released)
            next_section = findnext(startswith("## "), released, first_section + 1)
            return join(released[first_section:(something(next_section, length(released) + 1) - 1)], "\n")
        end
    end

    function check()
        for pkg in sort!(collect(keys(PACKAGES)))
            section = try
                preview(pkg, v"999.0.0")
            catch err
                err isa ErrorException || rethrow()
                error("$pkg ($(relpath(entry_dir(pkg), ROOT))): $(err.msg)")
            end
            println("$pkg: changelog entries are valid. Release section preview:\n")
            println(section, "\n")
        end
        return
    end

    function check_pr(pr::Int, added_files)
        expected = [relpath_entry(pkg, "$pr.md") for pkg in sort!(collect(keys(PACKAGES)))]
        foreign = filter(f -> f ∉ expected && basename(f) ∉ NON_ENTRY_FILES, added_files)
        isempty(foreign) ||
            error("This PR adds changelog files not named after its number $pr: $(join(foreign, ", ")). Rename them to `$pr.md`.")
        any(f -> isfile(joinpath(ROOT, f)), expected) ||
            error("Missing `$(expected[1])` (or `$(expected[2])`). Add it with one or more `- ` entries, or apply the `skip-changelog` label.")
        return check()
    end

    const USAGE = """
    Usage: julia tooling/changelog.jl <command>
      check                                         validate all entry files
      check-pr <PR number> [added files...]         CI check for a single PR
      release <version> [--package P] [--dry-run]   collect entries into CHANGELOG.md
                                                    (P = Drill (default) or DrillInterface;
                                                    --dry-run only prints the new section)"""

    function main(args)
        if length(args) == 1 && args[1] == "check"
            check()
        elseif length(args) >= 2 && args[1] == "check-pr"
            check_pr(parse(Int, args[2]), args[3:end])
        elseif length(args) >= 2 && args[1] == "release"
            version = VersionNumber(args[2])
            rest = args[3:end]
            dry_run = "--dry-run" in rest
            filter!(!=("--dry-run"), rest)
            pkg = "Drill"
            if length(rest) == 2 && rest[1] == "--package"
                pkg = rest[2]
            elseif !isempty(rest)
                error(USAGE)
            end
            if dry_run
                println(preview(pkg, version))
            else
                release!(version; changelog_path = changelog_path(pkg), dir = entry_dir(pkg))
                println("Added $version section to $(relpath(changelog_path(pkg), ROOT)) and removed the entry files.")
            end
        else
            error(USAGE)
        end
        return
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    Changelog.main(ARGS)
end
