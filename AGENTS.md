# AGENTS.md

## Cursor Cloud specific instructions

This is a pure Julia package (**Drill.jl** — a Deep Reinforcement Learning library). No Docker, Node, or Python services are required.

### Prerequisites

- **Julia ≥ 1.10** (installed via `juliaup` at `/home/ubuntu/.juliaup/bin`). Ensure `PATH` includes this directory.
- **Runic.jl** is installed in the global Julia environment for formatting/lint checks.

### Key commands

| Task | Command |
|---|---|
| Instantiate deps | `julia --project=. -e 'using Pkg; Pkg.instantiate()'` |
| Run tests | `julia --project=. -e 'using Pkg; Pkg.test()'` |
| Lint (Runic) | `julia -e 'using Runic; exit(Runic.main(["--check", "."]))'` |
| Format (Runic) | `julia -e 'using Runic; exit(Runic.main(["."]))'` |

### Non-obvious caveats

- **Test duration**: The full test suite (`Pkg.test()`) takes ~10 minutes due to heavy compilation (Enzyme, Mooncake, JET, Zygote). CI has a 60-minute timeout.
- **Test environment**: Test dependencies (Zygote, ClassicControlEnvironments, etc.) live in `test/Project.toml` with a `[sources]` section pointing Drill to `..`. To use test deps interactively, run `julia --project=test` and instantiate that environment separately.
- **JET false positives**: The test suite filters out `accumulated_data` warnings from `TimerOutputs.@timeit` — these are known false positives, not real issues.
- **AD backends filter**: By default, `@run_package_tests` filters out `:ad_backends` tagged tests (see `test/runtests.jl`). To run those too, modify the filter.
- **ClassicControlEnvironments.jl**: Fetched from GitHub as a git dependency in `test/Project.toml`. Requires network access during `Pkg.instantiate()` / `Pkg.test()`.
- **Wandb extension**: The Wandb test creates a CondaPkg environment under `test/.CondaPkg/` on first run — this is expected and can take extra time.
