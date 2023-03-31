using Documenter, Flux, NNlib, Functors, MLUtils, BSON, Optimisers, OneHotArrays, Zygote, ChainRulesCore, Plots, MLDatasets, Statistics, DataFrames

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
            "Quick Start" => "models/quickstart.md",
            "Fitting a Line" => "models/overview.md",
            "Gradients and Layers" => "models/basics.md",
            "Training" => "training/training.md",
            "Recurrence" => "models/recurrence.md",
            "GPU Support" => "gpu.md",
            "Saving & Loading" => "saving.md",
            "Performance Tips" => "performance.md",
        ],
        "Ecosystem" => "ecosystem.md",
        "Reference" => [
        # This essentially collects docstrings, with a bit of introduction.
            "Built-in Layers" => "models/layers.md",
            "Activation Functions" => "models/activation.md",
            "Weight Initialisation" => "utilities.md",
            "Loss Functions" => "models/losses.md",
            "Training API" => "training/reference.md",
            "Optimisation Rules" => "training/optimisers.md",
            "Shape Inference" => "outputsize.md",
            "Flat vs. Nested" => "destructure.md",
            "Callback Helpers" => "training/callbacks.md",
            "Gradients -- Zygote.jl" => "training/zygote.md",
            "Batching Data -- MLUtils.jl" => "data/mlutils.md",
            "OneHotArrays.jl" => "data/onehot.md",
            "Low-level Operations -- NNlib.jl" => "models/nnlib.md",
            "Nested Structures -- Functors.jl" => "models/functors.md",
         ],
        "Tutorials" => [
        # These walk you through various tasks. It's fine if they overlap quite a lot.
        # All the website tutorials can move here, perhaps much of the model zoo too?
        # Or perhaps those should just be trashed, model zoo versions are newer & more useful.
            "Linear Regression" => "tutorials/linear_regression.md",
            #=
            "Julia & Flux: 60 Minute Blitz" => "tutorials/2020-09-15-deep-learning-flux.md",
            "Multi-layer Perceptron" => "tutorials/2021-01-26-mlp.md",
            "Simple ConvNet" => "tutorials/2021-02-07-convnet.md",
            "Generative Adversarial Net" => "tutorials/2021-10-14-vanilla-gan.md",
            "Deep Convolutional GAN" => "tutorials/2021-10-08-dcgan-mnist.md",
            =#
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
    doctest = false, 
    # linkcheck = true,
    checkdocs = :exports,
    # strict = true,
    # strict = [
    #     :cross_references,
    #     :missing_docs,
    #     :doctest,
    #     :linkcheck,
    #     :parse_error,
    #     :example_block,
    #     :autodocs_block, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :setup_block
    # ],
)

doctest(Flux) # only test Flux modules

deploydocs(
    repo = "github.com/FluxML/Flux.jl.git",
    target = "build",
    push_preview = true
)
