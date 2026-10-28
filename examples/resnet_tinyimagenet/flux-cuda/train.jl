# Train the ResNet-18 with Flux on CUDA.jl: each training step runs eagerly, differentiated by
# Zygote. The model and training loop are in ../training.jl, and the data pipeline in ../data.jl.

include("../training.jl")

using CUDA, cuDNN

main(; kws...) = train(gpu_device(; force=true); eval_gc=true, kws...)

function (@main)(args)
    main(; parse_cli(args; description="Train a ResNet-18 on Tiny-ImageNet with Flux on CUDA.jl.")...)
    return nothing
end
