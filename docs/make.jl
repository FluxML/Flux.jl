using Documenter, Flux, NNlib, Functors, MLUtils, BSON, Optimisers, OneHotArrays, Zygote, ChainRulesCore, Plots, MLDatasets, Statistics, DataFrames, CUDA


DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive = true)

makedocs(
    modules = [Flux, NNlib, Functors, MLUtils, BSON, Optimisers, OneHotArrays, Zygote, ChainRulesCore, Base, Plots, MLDatasets, Statistics, DataFrames, CUDA],
    doctest = false,
    sitename = "Flux",
    # strict = [:cross_references,],
    pages = [
        "Welcome" => "index.md",
        "Guide" => [
        # You could read this end-to-end, or skip to what you need.
        # Aim is to cover each new concept exactly once (but not list all variants).
        # Hard to invent further divisions which aren't more confusing than helpful?
            "Quick Start" => "models/quickstart.md",
            "Fitting a Line" => "models/overview.md",
            "Gradients and Layers" => "models/basics.md",
            "Training" => "training/training.md",
            # "Regularisation" => "models/regularisation.md",  # consolidated in #2114
            "Recurrence" => "models/recurrence.md",
            "GPU Support" => "gpu.md",
            "Saving & Loading" => "saving.md",
            "Performance Tips" => "performance.md",
        ],
        "Reference" => [
        # This essentially collects docstrings, with a bit of introduction.
        # Probably the 📚 marker can be removed now.
            "Built-in Layers 📚" => "models/layers.md",
            "Activation Functions 📚" => "models/activation.md",
            "Weight Initialisation 📚" => "utilities.md",
            "Loss Functions 📚" => "models/losses.md",
            "Optimisation Rules 📚" => "training/optimisers.md",  # TODO move optimiser intro up to Training
            "Shape Inference 📚" => "outputsize.md",
            "Flat vs. Nested 📚" => "destructure.md",
            "Callback Helpers 📚" => "training/callbacks.md",
            "CUDA.jl 📚 (`cu`, `CuIterator`, ...)" => "reference/CUDA.md",  # not sure
            "NNlib.jl 📚 (`softmax`, `conv`, ...)" => "models/nnlib.md",
            "Zygote.jl 📚 (`gradient`, ...)" => "training/zygote.md",
            "MLUtils.jl 📚 (`DataLoader`, ...)" => "data/mlutils.md",
            "Functors.jl 📚 (`fmap`, ...)" => "models/functors.md",
            "OneHotArrays.jl 📚 (`onehot`, ...)" => "data/onehot.md",
         ],
        "Flux's Ecosystem" => "ecosystem.md",  # This is a links page
        "Tutorials" => [
        # These walk you through various tasks. It's fine if they overlap quite a lot.
        # All the website tutorials can move here, perhaps much of the model zoo too.
            "Linear Regression" => "tutorials/linear_regression.md",
            "Julia & Flux: 60 Minute Blitz" => "tutorials/2020-09-15-deep-learning-flux.md",
            "Multi-layer Perceptron" => "tutorials/2021-01-26-mlp.md",
            "Simple ConvNet" => "tutorials/2021-02-07-convnet.md",
            "Generative Adversarial Net" => "tutorials/2021-10-14-vanilla-gan.md",
            "Deep Convolutional GAN" => "tutorials/2021-10-08-dcgan-mnist.md",
            # Not really sure where this belongs... some in Fluxperimental, aim to delete?
            "Custom Layers" => "models/advanced.md",  # TODO move freezing to Training
        ],
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
