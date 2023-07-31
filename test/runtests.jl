using Flux
using Flux: OneHotArray, OneHotMatrix, OneHotVector
using Flux: params
using Test
using Random, Statistics, LinearAlgebra
using IterTools: ncycle
using Zygote

# ENV["FLUX_TEST_AMDGPU"] = "true"
ENV["FLUX_TEST_CUDA"] = "true"
# ENV["FLUX_TEST_METAL"] = "true"

include("test_utils.jl")

Random.seed!(0)

@testset verbose=true "Flux.jl" begin

  @testset "Utils" begin
    include("utils.jl")
  end

  @testset "Loading" begin
    include("loading.jl")
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

  @testset "functors" begin
    include("functors.jl")
  end

  if get(ENV, "FLUX_TEST_CUDA", "false") == "true"
    using CUDA, cuDNN
    Flux.gpu_backend!("CUDA")

    @test typeof(Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["CUDA"]]) <: Flux.FluxCUDADevice
    device = Flux.get_device()

    @testset "CUDA" begin
      if CUDA.functional()
        @test typeof(Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["CUDA"]].deviceID) <: CUDA.CuDevice
        @test typeof(device) <: Flux.FluxCUDADevice
        @test typeof(device.deviceID) <: CUDA.CuDevice
        @test Flux._get_device_name(device) in Flux.supported_devices()

        @info "Testing CUDA Support"
        include("ext_cuda/runtests.jl")
      else
        @warn "CUDA.jl package is not functional. Skipping CUDA tests."
        @test typeof(Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["CUDA"]].deviceID) <: Nothing 
      end
    end
  else
    @info "Skipping CUDA tests, set FLUX_TEST_CUDA=true to run them."
  end

  if get(ENV, "FLUX_TEST_AMDGPU", "false") == "true"
    using AMDGPU
    Flux.gpu_backend!("AMD")

    @test typeof(Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["AMD"]]) <: Flux.FluxAMDDevice
    device = Flux.get_device()

    if AMDGPU.functional() && AMDGPU.functional(:MIOpen)
      @test typeof(Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["AMD"]].deviceID) <: AMDGPU.HIPDevice 
      @test typeof(device) <: Flux.FluxAMDDevice
      @test typeof(device.deviceID) <: AMDGPU.HIPDevice
      @test Flux._get_device_name(device) in Flux.supported_devices()

      @testset "AMDGPU" begin
        include("ext_amdgpu/runtests.jl")
      end
    else
      @info "AMDGPU.jl package is not functional. Skipping AMDGPU tests."
      @test typeof(Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["AMD"]].deviceID) <: nothing 
    end
  else
    @info "Skipping AMDGPU tests, set FLUX_TEST_AMDGPU=true to run them."
  end

  if get(ENV, "FLUX_TEST_METAL", "false") == "true"
    using Metal
    Flux.gpu_backend!("Metal")

    @test typeof(Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["Metal"]]) <: Flux.FluxMetalDevice
    device = Flux.get_device()

    if Metal.functional()
      @test typeof(Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["Metal"]].deviceID) <: Metal.MTLDevice
      @test typeof(device) <: Flux.FluxMetalDevice
      @test typeof(device.deviceID) <: Metal.MTLDevice
      @test Flux._get_device_name(device) in Flux.supported_devices()

      @testset "Metal" begin
        include("ext_metal/runtests.jl")
      end
    else
      @info "Metal.jl package is not functional. Skipping Metal tests."
      @test typeof(Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["Metal"]].deviceID) <: Nothing 
    end
  else
    @info "Skipping Metal tests, set FLUX_TEST_METAL=true to run them."
  end

  @static if VERSION == v"1.9"
    using Documenter
    @testset "Docs" begin
      DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive=true)
      doctest(Flux)
    end
  end
end
