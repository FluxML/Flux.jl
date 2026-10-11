module FluxCUDAExt

# Load cuDNN, if installed, so that `using CUDA` is enough to obtain full CUDA/cuDNN support in Flux.
# This runs at load time rather than at the top level: Julia ≥ 1.13 precompile workers cannot load
# an undeclared dependency whose cache is not built yet, which left cuDNN silently out of the cache.
function __init__()
    ccall(:jl_generating_output, Cint, ()) == 1 && return
    try
        Base.require(Main, :cuDNN)
    catch
        @warn """Package cuDNN not found in current path.
        - Run `import Pkg; Pkg.add(\"cuDNN\")` to install the cuDNN package, then restart julia.
        - If cuDNN is not installed, some Flux functionalities will not be available when running on the GPU.
        """
    end
end

end
