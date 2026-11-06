# Train the ResNet-18 with Flux on CUDA.jl: each training step runs eagerly, differentiated by
# Zygote. The model and training loop are in ../training.jl, and the data pipeline in ../data.jl.

include("../training.jl")

using CUDA, cuDNN

# Time the cuDNN plans of each new convolution shape and keep the fastest, like PyTorch's
# `torch.backends.cudnn.benchmark`, instead of the first plan proposed by cuDNN's heuristics.
cuDNN.autotune!(true)

main(; kws...) = train(gpu_device(; force=true); eval_gc=true, kws...)

function (@main)(args)
    main(; parse_cli(args; description="Train a ResNet-18 on Tiny-ImageNet with Flux on CUDA.jl.")...)
    return nothing
end
