using Documenter, Flux, NNlib, Functors, MLUtils, BSON, Optimisers, 
      OneHotArrays, Zygote, ChainRulesCore, Plots, MLDatasets, Statistics, 
      DataFrames, JLD2

DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive = true)

makedocs(
    modules = [Flux, NNlib, Functors, MLUtils, Zygote, OneHotArrays, Optimisers, ChainRulesCore],
    sitename = "Flux",
    pages = [
        "Welcome" => "index.md",
        "Guide" => [
        # You could read this end-to-end, or skip to what you need.
        # Aim is to cover each new concept exactly once (but not list all variants).
        # Hard to invent further divisions which aren't more confusing than helpful?
            "Quick Start" => "guide/models/quickstart.md",
            "Fitting a Line" => "guide/models/overview.md",
            "Gradients and Layers" => "guide/models/basics.md",
            "Custom Layers" => "guide/models/custom_layers.md",
            "Training" => "guide/training/training.md",
            "Recurrence" => "guide/models/recurrence.md",
            "GPU Support" => "guide/gpu.md",
            "Saving & Loading" => "guide/saving.md",
            "Performance Tips" => "guide/performance.md",
        ],
        "Ecosystem" => "ecosystem.md",
        "Reference" => [
        # This essentially collects docstrings, with a bit of introduction.
            "Built-in Layers" => "reference/models/layers.md",
            "Activation Functions" => "reference/models/activation.md",
            "Weight Initialisation" => "reference/utilities.md",
            "Loss Functions" => "reference/models/losses.md",
            "Training API" => "reference/training/reference.md",
            "Optimisation Rules" => "reference/training/optimisers.md",
            "Shape Inference" => "reference/outputsize.md",
            "Flat vs. Nested" => "reference/destructure.md",
            "Callback Helpers" => "reference/training/callbacks.md",
            "Gradients -- Zygote.jl" => "reference/training/zygote.md",
            "Batching Data -- MLUtils.jl" => "reference/data/mlutils.md",
            "OneHotArrays.jl" => "reference/data/onehot.md",
            "Low-level Operations -- NNlib.jl" => "reference/models/nnlib.md",
            "Nested Structures -- Functors.jl" => "reference/models/functors.md",
            "Advanced" => "reference/misc-model-tweaking.md"
         ],
        "Tutorials" => [
        # These walk you through various tasks. It's fine if they overlap quite a lot.
        # All the website tutorials can move here, perhaps much of the model zoo too?
        # Or perhaps those should just be trashed, model zoo versions are newer & more useful.
            "Linear Regression" => "tutorials/linear_regression.md",
            "Logistic Regression" => "tutorials/logistic_regression.md",
            "Model Zoo" => "tutorials/model_zoo.md",
            #=
            # "Multi-layer Perceptron" => "tutorials/mlp.md",
            # "Julia & Flux: 60 Minute Blitz" => "tutorials/blitz.md",
            "Simple ConvNet" => "tutorials/2021-02-07-convnet.md",
            "Generative Adversarial Net" => "tutorials/2021-10-14-vanilla-gan.md",
            "Deep Convolutional GAN" => "tutorials/2021-10-08-dcgan-mnist.md",
            =#
        ],
    ],
    format = Documenter.HTML(
        sidebar_sitename = false,
        analytics = "UA-36890222-9",
        assets = ["assets/flux.css"],
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    doctest = false,   # done later
    checkdocs = :none, # :exports # Do not check if all functions appear in the docs
                                  # since it considers all packages
    warnonly = [:cross_references]
)

doctest(Flux) # only test Flux modules

deploydocs(
    repo = "github.com/FluxML/Flux.jl.git",
    target = "build",
    push_preview = true
)
