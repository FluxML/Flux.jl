# Train the ResNet-18 with Flux, compiling the training step with Reactant (XLA). The model and
# training loop are in ../training.jl, and the data pipeline in ../data.jl.

# Read when Reactant initializes: hides the INFO logs of XLA's autotuning.
get!(ENV, "TF_CPP_MIN_LOG_LEVEL", "1")

include("../training.jl")

using Reactant

# Each `Flux.trainstep!` on the Reactant model runs a compiled XLA program: the first call compiles
# the whole step (forward, backward and optimizer update), the next ones reuse it.
main(; kws...) = train(reactant_device(); compile=Reactant.compile, kws...)

function (@main)(args)
    main(; parse_cli(args; description="Train a ResNet-18 on Tiny-ImageNet with Flux and Reactant.")...)
    return nothing
end
