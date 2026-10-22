# `julia --project=. hellaswag.jl <checkpoint.jld2>`: the HellaSwag accuracy of a checkpoint, as in
# ../flux-reactant/hellaswag.jl, on CUDA.

using PythonCall   # before Flux, see ../flux-reactant/train.jl

include("../flux-reactant/model.jl")
include("../flux-reactant/data.jl")
include("../flux-reactant/hellaswag.jl")

using CUDA, cuDNN

hellaswag_checkpoint(ARGS[1]; device=gpu_device(; force=true))
