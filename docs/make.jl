using Documenter, Flux, NNlib, Functors, MLUtils, BSON, Optimisers, OneHotArrays, Zygote, ChainRulesCore


DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive = true)

makedocs(
    modules = [Flux, NNlib, Functors, MLUtils, BSON, Optimisers, OneHotArrays, Zygote, ChainRulesCore, Base],
    doctest = false,
    sitename = "Flux",
    # strict = [:cross_references,],
    pages = [
        "Home" => "index.md",
        "Building Models" => [
            "Overview" => "models/overview.md",
            "Basics" => "models/basics.md",
            "Recurrence" => "models/recurrence.md",
            "Layer Reference" => "models/layers.md",
            "Loss Functions" => "models/losses.md",
            "Regularisation" => "models/regularisation.md",
            "Custom Layers" => "models/advanced.md",
            "NNlib.jl" => "models/nnlib.md",
            "Activation Functions" => "models/activation.md",
         ],
         "Handling Data" => [
             "MLUtils.jl" => "data/mlutils.md",
             "OneHotArrays.jl" => "data/onehot.md",
         ],
         "Training Models" => [
             "Optimisers" => "training/optimisers.md",
             "Training" => "training/training.md",
             "Callback Helpers" => "training/callbacks.md",
             "Zygote.jl" => "training/zygote.md",
         ],
         "GPU Support" => "gpu.md",
         "Model Tools" => [
             "Saving & Loading" => "saving.md",
             "Shape Inference" => "outputsize.md",
             "Weight Initialisation" => "utilities.md",
             "Functors.jl" => "models/functors.md",
         ],
         "Performance Tips" => "performance.md",
         "Flux's Ecosystem" => "ecosystem.md",
    ],
    format = Documenter.HTML(
        sidebar_sitename = false,
        analytics = "UA-36890222-9",
        assets = ["assets/flux.css"],
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
)

deploydocs(
    repo = "github.com/FluxML/Flux.jl.git",
    target = "build",
    push_preview = true
)
