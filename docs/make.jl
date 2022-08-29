using Documenter, Flux, NNlib, Functors, MLUtils, BSON, Optimisers, OneHotArrays


DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive = true)

makedocs(
    modules = [Flux, NNlib, Functors, MLUtils, BSON, Optimisers, OneHotArrays],
    doctest = false,
    sitename = "Flux",
    strict = [:cross_references,],
    pages = [
        "Home" => "index.md",
        "Building Models" => [
            "Overview" => "models/overview.md",
            "Basics" => "models/basics.md",
            "Recurrence" => "models/recurrence.md",
            "Layer Reference" => "models/layers.md",
            "Loss Functions" => "models/losses.md",
            "Regularisation" => "models/regularisation.md",
            "Advanced Model Building" => "models/advanced.md",
            "NNlib.jl" => "models/nnlib.md",
            "Functors.jl" => "models/functors.md",
         ],
         "Handling Data" => [
             "MLUtils.jl" => "data/mlutils.md",
             "OneHotArrays.jl" => "data/onehot.md",
         ],
         "Training Models" => [
             "Optimisers" => "training/optimisers.md",
             "Training" => "training/training.md",
             "Zygote.jl" => "training/zygote.md",
         ],
         "GPU Support" => "gpu.md",
         "Model Tools" => [
             "Saving & Loading" => "saving.md",
             "Size Propagation" => "outputsize.md",
             "Weight Initialisation" => "utilities.md",
         ],
         "Performance Tips" => "performance.md",
         "Flux's Ecosystem" => "ecosystem.md",
    ],
    format = Documenter.HTML(
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
