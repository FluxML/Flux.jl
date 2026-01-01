using BSON: BSON
using FiniteDifferences: FiniteDifferences
using Flux
using Flux: OneHotArray, OneHotMatrix, OneHotVector, 
            onehotbatch, withgradient, pullback
using Flux.Losses: xlogx, xlogy
using Flux.Losses
using ForwardDiff: ForwardDiff
using Functors: Functors, fmapstructure_with_path
using IterTools: ncycle
using LinearAlgebra
using MLUtils: MLUtils, batch, unstack, unsqueeze, 
              unbatch, getobs, numobs, flatten, DataLoader
using Optimisers: Optimisers
using Pkg
using Random
using SparseArrays
using Statistics
using Test
using Zygote: Zygote
# const gradient = Flux.gradient  # both Flux & Zygote export this on 0.15
# const withgradient = Flux.withgradient

## Uncomment below to change the default test settings
# ENV["FLUX_TEST_AMDGPU"] = "true"
# ENV["FLUX_TEST_CUDA"] = "true"
# ENV["FLUX_TEST_METAL"] = "true"
# ENV["FLUX_TEST_CPU"] = "false"
# ENV["FLUX_TEST_DISTRIBUTED_MPI"] = "true"
# ENV["FLUX_TEST_DISTRIBUTED_NCCL"] = "true"
# ENV["FLUX_TEST_ENZYME"] = "false"
ENV["FLUX_TEST_REACTANT"] = "true"

const FLUX_TEST_ENZYME = get(ENV, "FLUX_TEST_ENZYME", VERSION < v"1.12-" ? "true" : "false") == "true"
const FLUX_TEST_CPU = get(ENV, "FLUX_TEST_CPU", "true") == "true"

# Reactant will automatically select a GPU backend, if available, and TPU backend, if available. 
# Otherwise it will fall back to CPU.
const FLUX_TEST_REACTANT = get(ENV, "FLUX_TEST_REACTANT", "true") == "true"

if FLUX_TEST_ENZYME || FLUX_TEST_REACTANT
  Pkg.add("Enzyme")
  using Enzyme: Enzyme, Const, Active, Duplicated
end

include("test_utils.jl") # for test_gradients

Random.seed!(0)

include("testsuite/normalization.jl")

function flux_testsuite(dev)
  @testset "Flux Test Suite" begin
    @testset "Normalization" begin
      normalization_testsuite(dev)
    end
  end
end

@testset verbose=true "Flux.jl" begin
  if FLUX_TEST_CPU
    flux_testsuite(cpu)

    @testset "Utils" begin
      include("utils.jl")
    end

    @testset "Loading" begin
      include("loading.jl")
    end

    @testset "Train" begin
      include("train.jl")
      include("tracker.jl")
    end

    @testset "Data" begin
      include("data.jl")
    end

    @testset "Losses" begin
      include("losses.jl")
      include("ctc.jl")
    end

    @testset "Layers" begin
      include("layers/attention.jl")
      include("layers/basic.jl")
      include("layers/normalisation.jl")
      include("layers/stateless.jl")
      include("layers/recurrent.jl")
      include("layers/conv.jl")
      include("layers/upsample.jl")
      include("layers/show.jl")
      include("layers/macro.jl")
    end

    @testset "outputsize" begin
      using Flux: outputsize
      include("outputsize.jl")
    end

    @testset "functors" begin
      include("functors.jl")
    end

    @testset "deprecations" begin
      include("deprecations.jl")
    end
  else
      @info "Skipping CPU tests."
  end

  if get(ENV, "FLUX_TEST_CUDA", "false") == "true"
    Pkg.add(["CUDA", "cuDNN"])
    # using CUDA, cuDNN
    using CUDA # cuDNN is loaded by FluxCUDAExt

    if CUDA.functional()
      @testset "CUDA" begin
        include("ext_cuda/runtests.jl")

        flux_testsuite(gpu)
      end
    else
      @warn "CUDA.jl package is not functional. Skipping CUDA tests."
    end
  else
    @info "Skipping CUDA tests, set FLUX_TEST_CUDA=true to run them."
  end

  if get(ENV, "FLUX_TEST_AMDGPU", "false") == "true"
    Pkg.add("AMDGPU")
    using AMDGPU

    if AMDGPU.functional() && AMDGPU.functional(:MIOpen)
      @testset "AMDGPU" begin
        include("ext_amdgpu/runtests.jl")

        flux_testsuite(gpu)
      end
    else
      @info "AMDGPU.jl package is not functional. Skipping AMDGPU tests."
    end
  else
    @info "Skipping AMDGPU tests, set FLUX_TEST_AMDGPU=true to run them."
  end

  if get(ENV, "FLUX_TEST_METAL", "false") == "true"
    Pkg.add("Metal")
    using Metal

    if Metal.functional()
      @testset "Metal" begin
        include("ext_metal/runtests.jl")

        flux_testsuite(gpu)
      end
    else
      @info "Metal.jl package is not functional. Skipping Metal tests."
    end
  else
    @info "Skipping Metal tests, set FLUX_TEST_METAL=true to run them."
  end

  if get(ENV, "FLUX_TEST_DISTRIBUTED_MPI", "false") == "true" || get(ENV, "FLUX_TEST_DISTRIBUTED_NCCL", "false") == true
    Pkg.add(["MPI"])
    using MPI

    if get(ENV, "FLUX_TEST_DISTRIBUTED_NCCL", "false") == "true"
      Pkg.add(["NCCL"])
      using NCCL
      import CUDA
    end

    @testset "Distributed" begin
      include("ext_distributed/runtests.jl")
    end

  else
    @info "Skipping Distributed tests, set FLUX_TEST_DISTRIBUTED_MPI or FLUX_TEST_DISTRIBUTED_NCCL=true to run them."
  end

  if FLUX_TEST_ENZYME
    ## Pkg.add("Enzyme") is already done above
    @testset "Enzyme" begin
      if FLUX_TEST_CPU
        include("ext_enzyme/enzyme.jl")
      end
    end
  else
    @info "Skipping Enzyme tests, set FLUX_TEST_ENZYME=true to run them."
  end

  if FLUX_TEST_REACTANT
    ## This Pg.add has to be done after Pkg.add("CUDA") otherwise CUDA.jl
    ## will not be functional and complain with: 
    # ┌ Error: CUDA.jl could not find an appropriate CUDA runtime to use.
    # │ 
    # │ CUDA.jl's JLLs were precompiled without an NVIDIA driver present.
    Pkg.add("Reactant")
    using Reactant: Reactant
    @testset "Reactant" begin
      include("ext_reactant/test_utils_reactant.jl")
      include("ext_reactant/reactant.jl")
    end
  else
    @info "Skipping Reactant tests, set FLUX_TEST_REACTANT=true to run them."
  end

end
