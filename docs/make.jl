using Documenter, Flux

makedocs(modules=[Flux],
         doctest = false,
         format = :html,
         analytics = "UA-36890222-9",
         sitename = "Flux",
         assets = ["../flux.css"],
         pages = ["Home" => "index.md",
                  "Models" =>
                    ["Basics" => "models/basics.md",
                     "Recurrence" => "models/recurrence.md",
                     "Layers" => "models/layers.md"],
                  "Contributing & Help" => "contributing.md"])

deploydocs(
   repo = "github.com/FluxML/Flux.jl.git",
   modules = [Flux],
   target = "build",
   osname = "linux",
   julia = "0.6",
   deps = nothing,
   make = nothing)
