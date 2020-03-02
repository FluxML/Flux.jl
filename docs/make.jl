using Pkg;
Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()
Pkg.activate(); Pkg.instantiate()

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter, Flux, NNlib

makedocs(modules=[Flux, NNlib],
         sitename = "Flux",
         pages = ["Home" => "index.md",
                  "Building Models" =>
                    ["Basics" => "models/basics.md",
                     "Recurrence" => "models/recurrence.md",
                     "Regularisation" => "models/regularisation.md",
                     "Model Reference" => "models/layers.md",
                     "NNlib" => "models/nnlib.md"],
                  "Handling Data" =>
                    ["One-Hot Encoding" => "data/onehot.md",
                     "DataLoader" => "data/dataloader.md"],
                  "Training Models" =>
                    ["Optimisers" => "training/optimisers.md",
                     "Training" => "training/training.md"],
                  "GPU Support" => "gpu.md",
                  "Saving & Loading" => "saving.md",
                  "The Julia Ecosystem" => "ecosystem.md",
                  "Performance Tips" => "performance.md",
                  "Community" => "community.md"],
         format = Documenter.HTML(assets = ["assets/flux.css"],
                                  analytics = "UA-36890222-9",
                                  prettyurls = haskey(ENV, "CI")))

deploydocs(repo = "github.com/FluxML/Flux.jl.git")
