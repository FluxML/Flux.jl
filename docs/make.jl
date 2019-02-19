using Documenter, Flux, NNlib

makedocs(modules=[Flux, NNlib],
         doctest = true,
         analytics = "UA-36890222-9",
         sitename = "Flux",
         # Uncomment below for local build
         #format = Documenter.HTML(prettyurls = false),
         assets = ["assets/flux.css"],
         pages = ["Home" => "index.md",
                  "Building Models" =>
                    ["Basics" => "models/basics.md",
                     "Recurrence" => "models/recurrence.md",
                     "Regularisation" => "models/regularisation.md",
                     "Model Reference" => "models/layers.md"],
                  "Training Models" =>
                    ["Optimisers" => "training/optimisers.md",
                     "Training" => "training/training.md"],
                  "One-Hot Encoding" => "data/onehot.md",
                  "GPU Support" => "gpu.md",
                  "Saving & Loading" => "saving.md",
                  "Performance Tips" => "performance.md",
                  "Internals" =>
                    ["Backpropagation" => "internals/tracker.md"],
                  "Community" => "community.md"])

deploydocs(repo = "github.com/FluxML/Flux.jl.git")
