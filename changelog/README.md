# Unreleased changelog entries

Every PR with user-facing changes adds a file `changelog/<PR number>.md` containing one or more entries. Changes to the `DrillInterface` subpackage go in `DrillInterface/changelog/<PR number>.md` instead (a PR touching both packages adds both files).

```md
- Fixed `foo` erroring on empty input.
- Added the `bar` keyword to `baz`.
  Indented lines continue the entry above.
```

Entries can optionally be grouped under the sections `## Breaking`, `## Added`, `## Changed`, `## Deprecated`, `## Removed` and `## Fixed`:

```md
## Breaking

- Removed the deprecated `qux` keyword.

## Fixed

- Fixed `foo` erroring on empty input.
```

Each entry starts with `- `, indented lines continue the previous entry, and blank lines are ignored. Nothing else is allowed. Entries before the first heading are unsectioned. Do not add the PR link, it is appended automatically.

Commands (run from the repo root):

```bash
julia tooling/changelog.jl check                                   # validate all entries, print previews
julia tooling/changelog.jl release 0.1.0 --dry-run                  # print the would-be Drill section (registry release notes)
julia tooling/changelog.jl release 0.1.0                            # write it into CHANGELOG.md and delete the entry files
julia tooling/changelog.jl release 0.3.0 --package DrillInterface   # same for DrillInterface/CHANGELOG.md
```

The release command collects all entries, newest PR first, into a new `## [version] - date` section of `CHANGELOG.md`. Unsectioned entries come first, followed by one `###` subsection per section in the order listed above.

CI (`.github/workflows/Changelog.yml`) checks that a PR adds an entry file named after its own number unless the `skip-changelog` label is set, and validates all entries with `julia tooling/changelog.jl check`.

This setup is adapted from [Makie.jl's changelog system](https://github.com/MakieOrg/Makie.jl/tree/master/changelog) (MIT License).
