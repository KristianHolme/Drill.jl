# Building and previewing documentation locally

This folder uses [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) with [DocumenterVitepress.jl](https://github.com/LuxDL/DocumenterVitepress.jl). The build entry point is `make.jl`.

## Prerequisites

- A Julia installation and this repository checked out.
- **[LiveServer.jl](https://github.com/tlienart/LiveServer.jl)** installed in your **default (global) Julia environment** (not listed in `docs/Project.toml`). Julia **stacks** the default environment with the active project, so `using LiveServer` still works when you run Julia with `--project=docs`. Add it from a Julia REPL, for example:

  ```julia
  using Pkg
  Pkg.activate()   # default environment
  Pkg.add("LiveServer")
  ```

  Or in Pkg mode: `]` then `activate` (to ensure you are on the default env), then `add LiveServer`.

## Docs dependencies

From the **repository root** (parent of `docs/`), instantiate the `docs` environment once (or after `docs/Project.toml` changes):

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
```

## Build

From the **repository root**:

```bash
julia --project=docs docs/make.jl
```

The first run may take a while while Vitepress and dependencies run. For a normal local build, DocumenterVitepress writes the static site under **`docs/build/1`** (see upstream notes on numbered `build` subfolders).

## Preview with LiveServer

From the **repository root**, use LiveServer to serve the built site (you can use any julia environment where LiveServer is available. If installed in the global env, LiveServer is available from all other environments.

```julia
using LiveServer
LiveServer.serve(dir = "docs/build/1")
```

LiveServer prints a local URL (for example `http://127.0.0.1:8000`); open that in your browser.

## Edit loop

After changing files under `docs/src/`, run the build command again, then refresh the browser. LiveServer serves the files on disk; it does not rebuild Documenter/Vitepress for you.

## Note on `deploydocs`

`make.jl` ends with `DocumenterVitepress.deploydocs`, which is meant for CI deployment to `gh-pages`. On a typical machine without GitHub Actions deploy credentials, pushing may be skipped or fail, but the **built HTML under `docs/build/1`** is still what you use for local preview.
