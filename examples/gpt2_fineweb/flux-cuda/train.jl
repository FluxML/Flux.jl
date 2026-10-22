# Pretrain GPT-2 with Flux on CUDA.jl: each training step runs eagerly, differentiated by Zygote.
# The model, data, HellaSwag evaluation and training loop are those of the Reactant version in
# ../flux-reactant, which this script includes.

using PythonCall   # before Flux, see ../flux-reactant/train.jl

include("../flux-reactant/training.jl")

using CUDA, cuDNN

# `tf32` lets cuBLAS run Float32 matrix multiplications on TF32 tensor cores; `f32` and `bf16` keep
# CUDA.jl's default math mode, in which Float32 matrix multiplications are exact. CUDA.jl frees GPU
# memory when the garbage collector finalizes the arrays, hence `gc = true`. The backward pass
# keeps all the activations of the forward pass, so GPT-2 124M fits 8 sequences per micro-batch
# in 32 GB, half the preset's 16.
function main(; preset="tinystories", batchsize=(preset == "fineweb" ? 8 : nothing),
              precision="bf16", log_dir=joinpath(@__DIR__, "log"), kws...)
    CUDA.math_mode!(precision == "tf32" ? CUDA.FAST_MATH : CUDA.DEFAULT_MATH)
    return train(gpu_device(; force=true); gc=true, preset, batchsize, precision, log_dir, kws...)
end

function (@main)(args)
    main(; parse_cli(args; description="Pretrain GPT-2 with Flux on CUDA.")...)
    return nothing
end
