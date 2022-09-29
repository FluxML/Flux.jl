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
            "Built-in Layers 📚" => "models/layers.md",
            "Recurrence" => "models/recurrence.md",
            "Custom Layers" => "models/advanced.md",  # TODO move freezing to Training
            "Activation Functions 📚" => "models/activation.md",
            "NNlib.jl 📚 (`softmax`, `conv`, ...)" => "models/nnlib.md",
         ],
         "Handling Data" => [
             "MLUtils.jl 📚 (`DataLoader`, ...)" => "data/mlutils.md",
             "OneHotArrays.jl 📚 (`onehot`, ...)" => "data/onehot.md",
         ],
         "Training Models" => [
             "Training" => "training/training.md",
             "Regularisation" => "models/regularisation.md",
             "Loss Functions 📚" => "models/losses.md",
             "Optimisation Rules 📚" => "training/optimisers.md",  # TODO move optimiser intro up to Training
             "Callback Helpers 📚" => "training/callbacks.md",
             "Zygote.jl 📚 (`gradient`, ...)" => "training/zygote.md",
         ],
         "Model Tools" => [
             "GPU Support" => "gpu.md",
             "Saving & Loading" => "saving.md",
             "Shape Inference 📚" => "outputsize.md",
             "Weight Initialisation 📚" => "utilities.md",
             "Flat vs. Nested 📚" => "destructure.md",
             "Functors.jl 📚 (`fmap`, ...)" => "models/functors.md",
         ],
         "Performance Tips" => "performance.md",
         "Flux's Ecosystem" => "ecosystem.md",
         # "Tutorials" => [  # TODO, maybe
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
