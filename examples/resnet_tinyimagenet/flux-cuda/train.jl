# Train the ResNet-18 with Flux on CUDA.jl: each training step runs eagerly, differentiated by
# Zygote. The data pipeline, model and training loop are in ../training.jl.

include("../training.jl")

using CUDA, cuDNN

# CUDA.jl frees GPU memory when the garbage collector finalizes the arrays, hence `gc = true`.
main(; kws...) = train(gpu_device(; force=true); gc=true, kws...)

function (@main)(args)
    main(; parse_cli(args; description="Train a ResNet-18 on Tiny-ImageNet with Flux on CUDA.jl.")...)
    return nothing
end
