using Documenter, Flux, NNlib

DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive=true)
makedocs(modules=[Flux, NNlib],
         doctest = true,
         sitename = "Flux",
         format = Documenter.HTML(
                 analytics = "UA-36890222-9",
                 assets = ["assets/flux.css"],
                 prettyurls = get(ENV, "CI", nothing) == "true",
         ),
         pages = ["Home" => "index.md",
                  "Building Models" =>
                    ["Basics" => "models/basics.md",
                     "Recurrence" => "models/recurrence.md",
                     "Regularisation" => "models/regularisation.md",
                     "Model Reference" => "models/layers.md",
                     "Advanced Model Building" => "models/advanced.md",
                     "NNlib" => "models/nnlib.md"],
                  "Handling Data" =>
                    ["One-Hot Encoding" => "data/onehot.md",
                     "DataLoader" => "data/dataloader.md"],
                  "Training Models" =>
                    ["Optimisers" => "training/optimisers.md",
                     "Loss Functions" => "training/loss_functions.md",
                     "Training" => "training/training.md"],
                  "GPU Support" => "gpu.md",
                  "Saving & Loading" => "saving.md",
                  "The Julia Ecosystem" => "ecosystem.md",
                  "Utility Functions" => "utilities.md",
                  "Performance Tips" => "performance.md",
                  "Community" => "community.md"],
         )

deploydocs(repo = "github.com/FluxML/Flux.jl.git",
           target = "build",
           push_preview = true)
