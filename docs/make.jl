using Documenter, Flux

makedocs(modules=[Flux],
         doctest = false,
         format = :html,
         analytics = "UA-36890222-9",
         sitename = "Flux",
         assets = ["../flux.css"],
         pages = ["Home" => "index.md",
                  "Building Models" =>
                    ["Basics" => "models/basics.md",
                     "Recurrence" => "models/recurrence.md",
                     "Layer Reference" => "models/layers.md"],
                  "Training Models" =>
                    ["Optimisers" => "training/optimisers.md",
                     "Training" => "training/training.md"],
                  "One-Hot Encoding" => "data/onehot.md",
                  "GPU Support" => "gpu.md",
                  "Contributing & Help" => "contributing.md"])

deploydocs(
   repo = "github.com/FluxML/Flux.jl.git",
   target = "build",
   osname = "linux",
   julia = "0.6",
   deps = nothing,
   make = nothing)
