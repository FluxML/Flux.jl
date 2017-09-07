using Documenter, Flux

makedocs(modules=[Flux],
         doctest = false,
         format = :html,
         analytics = "UA-36890222-9",
         sitename = "Flux",
         assets = ["../flux.css"],
         pages = ["Home" => "index.md",
                  "Contributing & Help"   => "contributing.md"])

deploydocs(
   repo = "github.com/FluxML/Flux.jl.git",
   target = "build",
   osname = "linux",
   julia = "0.6",
   deps = nothing,
   make = nothing)
