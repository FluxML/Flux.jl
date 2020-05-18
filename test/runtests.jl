using Flux 
using Flux.Data
using Test 
using Random, Statistics, LinearAlgebra
using Documenter
using IterTools: ncycle

Random.seed!(0)

@testset "Flux" begin

  @testset "Utils" begin
    include("utils.jl")
  end

  @testset "Onehot" begin
    include("onehot.jl")
  end

  @testset "Optimise" begin
    include("optimise.jl")
  end

  @testset "Data" begin
    include("data.jl")
  end

  @testset "Layers" begin
    include("layers/basic.jl")
    include("layers/normalisation.jl")
    include("layers/stateless.jl")
    include("layers/conv.jl")
  end

  @testset "CUDA" begin
    if Flux.use_cuda[]
      include("cuda/cuda.jl")
    else
      @warn "CUDA unavailable, not testing GPU support"
    end
  end

  @testset "Docs" begin
    if VERSION >= v"1.4"
      DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive=true)
      doctest(Flux)
    end
  end

end # testset Flux
