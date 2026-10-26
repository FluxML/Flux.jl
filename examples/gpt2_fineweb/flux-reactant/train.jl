# Pretrain GPT-2 with Flux, compiling the training step with Reactant (XLA).
# The first run for a dataset streams and tokenizes it into `data_dir` (see data.jl).

# Read when Reactant initializes. XLA reserves 75% of the GPU memory by default; the INFO logs it
# would print include a register-spill note for every matmul kernel it autotunes.
get!(ENV, "XLA_REACTANT_GPU_MEM_FRACTION", "0.9")
get!(ENV, "TF_CPP_MIN_LOG_LEVEL", "1")

# Before Flux: CondaPkg resolves the Python environment when PythonCall loads, capping libstdc++ at
# Julia's, which Julia 1.13 fails to detect once CompilerSupportLibraries_jll is loaded
# (JuliaLang/julia#63967). Without the cap, the environment re-resolves whenever the load order changes.
using PythonCall

include("training.jl")   # the model, data, HellaSwag and training loop, shared with ../flux-cuda

using Reactant
using Reactant: PrecisionConfig

# `tf32` is XLA's default precision for Float32 matrix multiplications on NVIDIA GPUs, and `f32`
# asks for the highest. Reactant reads the configuration when it compiles a function, so it wraps
# the whole training loop.
precision_config(precision) = precision == "f32" ? (; dot_general_precision=PrecisionConfig.HIGHEST) : (;)

# Each `Flux.trainstep!` on the Reactant model runs a compiled XLA program: the first call compiles
# the whole step (forward, backward and optimizer update), the next ones reuse it.
function main(; precision="bf16", log_dir=joinpath(@__DIR__, "log"), kws...)
    Reactant.with_config(; precision_config(precision)...) do
        train(reactant_device(); compile=Reactant.compile, precision, log_dir, kws...)
    end
end

function (@main)(args)
    main(; parse_cli(args; description="Pretrain GPT-2 with Flux and Reactant.")...)
    return nothing
end
