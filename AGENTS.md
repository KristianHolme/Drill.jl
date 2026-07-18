# Agent notes for Drill.jl

## Runic formatting

All Julia source must pass [Runic.jl](https://github.com/fredrikekre/Runic.jl) formatting. CI runs `.github/workflows/Runic.yml` on pull requests.

Format changed files before pushing:

```bash
julia -e 'using Runic; exit(Runic.main(String["--inplace", "path/to/file.jl"]))'
# or a directory:
julia -e 'using Runic; exit(Runic.main(String["--inplace", "src", "ext", "test", "examples", "benchmark"]))'
```

Check without writing:

```bash
julia -e 'using Runic; exit(Runic.main(String["--check", "src", "ext"]))'
```

Notable Runic rules for this repo:

- Explicit `return` on the last expression in `function` / `macro` bodies
- Four-space indentation
- Spaces around operators and keywords (`f(a = 1)`, `x = 1`, `for i in 1:n`)
- No trailing blank line at end of file beyond the single final newline
- Multiline call / array / tuple literals use a leading and trailing newline with trailing commas for arrays/tuples

## Explicit imports

All modules (including package extensions under `ext/`) must use **explicit imports**. CI enforces this via ExplicitImports.jl in `test/test_quality.jl`.

Do **not** write:

```julia
using Foo
using Foo: bar   # still pulls Foo into scope implicitly if you also bare-`using Foo`
f() = bar() + Foo.baz()
```

Prefer:

```julia
using Foo: Foo, bar, baz
# or
import Foo: bar, baz
```

For package extensions, qualify only what you need from Drill and the weak dependency:

```julia
module Drill_SomeExt

using Drill: Drill, SomeType, some_helper
using SomeDep: SomeDep, some_fun
import Drill: some_api_to_extend

function some_api_to_extend(x::SomeType)
    return some_fun(some_helper(x))
end

end
```

After adding or editing an extension, load it in a test session (add the weakdep to `test/Project.toml` if needed) and confirm:

```julia
using ExplicitImports, Drill, PrettyTables  # example weakdep
ext = Base.get_extension(Drill, :Drill_PrettyTablesExt)
@assert check_no_implicit_imports(ext) === nothing
@assert check_no_stale_explicit_imports(ext) === nothing
```

`check_no_implicit_imports(Drill)` also analyzes **loaded** extensions, so a weakdep present in the test environment can cause quality tests to fail if that extension uses bare `using`.
