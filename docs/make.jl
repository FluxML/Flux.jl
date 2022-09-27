using Documenter, Flux, NNlib, Functors, MLUtils, BSON, Optimisers, OneHotArrays, Zygote, ChainRulesCore


DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive = true)

makedocs(
    modules = [Flux, NNlib, Functors, MLUtils, BSON, Optimisers, OneHotArrays, Zygote, ChainRulesCore, Base],
    doctest = false,
    sitename = "Flux",
    # strict = [:cross_references,],
    pages = [
        "Getting Started" => [
            "Welcome" => "index.md",
            "Quick Start" => "models/quickstart.md",
            "Fitting a Line" => "models/overview.md",
            "Gradients and Layers" => "models/basics.md",
        ],
        "Building Models" => [
            "Built-in Layers" => "models/layers.md",
            "Loss Functions" => "models/losses.md",
            "Recurrence" => "models/recurrence.md",
            "Regularisation" => "models/regularisation.md",
            "Custom Layers" => "models/advanced.md",
            "Activation Functions" => "models/activation.md",
            "NNlib.jl" => "models/nnlib.md",
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
         "Model Tools" => [
             "GPU Support" => "gpu.md",
             "Saving & Loading" => "saving.md",
             "Shape Inference" => "outputsize.md",
             "Weight Initialisation" => "utilities.md",
             "Functors.jl" => "models/functors.md",
         ],
         "Performance Tips" => "performance.md",
         "Flux's Ecosystem" => "ecosystem.md",
         # "Tutorials" => [
         #
         # ],
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

#=
] activate .
instantiate

time julia --project --color=yes make.jl
=#
