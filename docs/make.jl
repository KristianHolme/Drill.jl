using Documenter
using DocumenterVitepress
using Drill

makedocs(;
    sitename = "Drill.jl",
    authors = "Kristian Holme",
    modules = [Drill],
    warnonly = true,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/KristianHolme/Drill.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Guide" => [
            "Environments" => "guide/environments.md",
            "Algorithms" => "guide/algorithms.md",
            "Wrappers" => "guide/wrappers.md",
        ],
        "Examples" => [
            "Logging" => "examples/logging.md",
        ],
        "API Reference" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/KristianHolme/Drill.jl.git",
    devbranch = "main",
    push_preview = true,
)
