using Documenter, Flux, NNlib, Functors, MLUtils

DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive = true)

makedocs(modules = [Flux, NNlib, Functors, MLUtils],
         doctest = false,
         sitename = "Flux",
         pages = ["Home" => "index.md",
                  "Building Models" =>
                    ["Overview" => "models/overview.md",
                     "Basics" => "models/basics.md",
                     "Recurrence" => "models/recurrence.md",
                     "Model Reference" => "models/layers.md",
                     "Loss Functions" => "models/losses.md",
                     "Regularisation" => "models/regularisation.md",
                     "Advanced Model Building" => "models/advanced.md",
                     "NNlib" => "models/nnlib.md",
                     "Functors" => "models/functors.md"],
                  "Handling Data" =>
                    ["One-Hot Encoding" => "data/onehot.md",
                     "MLUtils" => "data/mlutils.md"],
                  "Training Models" =>
                    ["Optimisers" => "training/optimisers.md",
                     "Training" => "training/training.md"],
                  "GPU Support" => "gpu.md",
                  "Saving & Loading" => "saving.md",
                  "The Julia Ecosystem" => "ecosystem.md",
                  "Utility Functions" => "utilities.md",
                  "Performance Tips" => "performance.md",
                  "Datasets" => "datasets.md",
                  "Community" => "community.md"],
         format = Documenter.HTML(
             analytics = "UA-36890222-9",
             assets = ["assets/flux.css"],
             prettyurls = get(ENV, "CI", nothing) == "true"),
         )

deploydocs(repo = "github.com/FluxML/Flux.jl.git",
           target = "build",
           push_preview = true)
