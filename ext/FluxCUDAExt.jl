module FluxCUDAExt
try
    # Let's try to load the cuDNN package if it is not already loaded
    # Thanks to this, users can just write `using CUDA`
    # to obtain full CUDA/cuDNN support in Flux. 
    Base.require(Main, :cuDNN)
catch
    @warn """Package cuDNN not found in current path.
    - Run `import Pkg; Pkg.add(\"cuDNN\")` to install the cuDNN package, then restart julia.
    - If cuDNN is not installed, some Flux functionalities will not be available when running on the GPU.
    """
end

end 