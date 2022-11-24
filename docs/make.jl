using Documenter, Flux, NNlib, Functors, MLUtils, BSON, Optimisers, OneHotArrays, Zygote, ChainRulesCore, Plots, MLDatasets, Statistics, DataFrames


DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive = true)

makedocs(
    modules = [Flux, NNlib, Functors, MLUtils, BSON, Optimisers, OneHotArrays, Zygote, ChainRulesCore, Base, Plots, MLDatasets, Statistics, DataFrames],
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
         "Tutorials" => [
             # Roughly in order of increasing complexity? Not chronological.
            "Linear Regression" => "tutorials/linear_regression.md",
            "Julia & Flux: 60 Minute Blitz" => "tutorials/2020-09-15-deep-learning-flux.md",
            "Multi-layer Perceptron" => "tutorials/2021-01-26-mlp.md",
            "Simple ConvNet" => "tutorials/2021-02-07-convnet.md",
            "Generative Adversarial Net" => "tutorials/2021-10-14-vanilla-gan.md",
            "Deep Convolutional GAN" => "tutorials/2021-10-08-dcgan-mnist.md",
            # Not really sure where this belongs... some in Fluxperimental, aim to delete?
            "Custom Layers" => "models/advanced.md",  # TODO move freezing to Training
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
