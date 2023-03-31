using Flux
using Flux: OneHotArray, OneHotMatrix, OneHotVector
using Flux: params
using Test
using Random, Statistics, LinearAlgebra
using IterTools: ncycle
using Zygote
using CUDA

include("test_utils.jl")

Random.seed!(0)

@testset verbose=true "Flux.jl" begin

  @testset "Utils" begin
    include("utils.jl")
  end

  @testset "Optimise / Train" begin
    include("optimise.jl")
    include("train.jl")
  end

  @testset "Data" begin
    include("data.jl")
  end

  @testset "Losses" begin
    include("losses.jl")
    include("ctc.jl")
    CUDA.functional() && include("ctc-gpu.jl")
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
  end

  @testset "outputsize" begin
    using Flux: outputsize
    include("outputsize.jl")
  end

  @testset "CUDA" begin
    if CUDA.functional()
      include("cuda/runtests.jl")
    else
      @warn "CUDA unavailable, not testing GPU support"
    end
  end

  @static if VERSION == v"1.6"
    using Documenter
    @testset "Docs" begin
      DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive=true)
      doctest(Flux)
    end
  end

  if get(ENV, "FLUX_TEST_AMDGPU", "false") == "true"
    using AMDGPU
    AMDGPU.versioninfo()
    if AMDGPU.functional() && AMDGPU.functional(:MIOpen)
      @show AMDGPU.MIOpen.version()
      @testset "AMDGPU" begin
        include("amd/runtests.jl")
      end
    else
      @info "AMDGPU.jl package is not functional. Skipping AMDGPU tests."
    end
  else
    @info "Skipping AMDGPU tests, set FLUX_TEST_AMDGPU=true to run them."
  end
end
