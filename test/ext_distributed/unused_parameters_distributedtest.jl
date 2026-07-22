using Test
using Flux
using Flux: DistributedUtils, MPIBackend, NCCLBackend

backend_string = ARGS[1]

if backend_string == "mpi"
    import MPI
elseif backend_string == "nccl"
    import MPI, NCCL, CUDA
else
    error("unsupported backend: $backend_string")
end

const btype = backend_string == "nccl" ? NCCLBackend : MPIBackend

DistributedUtils.initialize(btype)
backend = DistributedUtils.get_distributed_backend(btype)

@testset "resolve_unused_parameters!!" begin
    # Test 1: Replaces nothing gradients with zero-filled arrays
    @testset "replaces nothing with zeros" begin
        model = (a = Float32[1.0, 2.0, 3.0], b = Float32[4.0, 5.0])
        gs = (a = nothing, b = Float32[0.1, 0.2])

        gs_resolved = DistributedUtils.resolve_unused_parameters!!(backend, gs, model)

        @test gs_resolved.a isa Array{Float32}
        @test size(gs_resolved.a) == size(model.a)
        @test all(gs_resolved.a .== 0)
        @test gs_resolved.b == Float32[0.1, 0.2]
    end

    # Test 2: Handles nested parameter trees (NamedTuples)
    @testset "nested NamedTuples" begin
        model = (layer1 = (weight = randn(Float32, 3, 2), bias = randn(Float32, 3)),
                 layer2 = (weight = randn(Float32, 1, 3), bias = randn(Float32, 1)))
        gs = (layer1 = nothing,
              layer2 = (weight = ones(Float32, 1, 3), bias = ones(Float32, 1)))

        gs_resolved = DistributedUtils.resolve_unused_parameters!!(backend, gs, model)

        @test gs_resolved.layer1.weight isa Array{Float32}
        @test size(gs_resolved.layer1.weight) == (3, 2)
        @test all(gs_resolved.layer1.weight .== 0)
        @test gs_resolved.layer1.bias isa Array{Float32}
        @test size(gs_resolved.layer1.bias) == (3,)
        @test all(gs_resolved.layer1.bias .== 0)
        @test gs_resolved.layer2.weight == ones(Float32, 1, 3)
        @test gs_resolved.layer2.bias == ones(Float32, 1)
    end

    # Test 3: No-op when all gradients are present
    @testset "no-op when all present" begin
        model = (a = randn(Float32, 4), b = randn(Float32, 2, 3))
        gs_orig = (a = ones(Float32, 4), b = ones(Float32, 2, 3))

        gs_resolved = DistributedUtils.resolve_unused_parameters!!(backend, gs_orig, model)

        @test gs_resolved.a == gs_orig.a
        @test gs_resolved.b == gs_orig.b
    end

    # Test 4: Preserves exact shapes and element types
    @testset "preserves shapes and types" begin
        model = (w1 = randn(Float32, 5, 3),
                 w2 = randn(Float64, 2, 4),
                 w3 = randn(Float32, 7))
        gs = (w1 = nothing, w2 = nothing, w3 = randn(Float32, 7))

        gs_resolved = DistributedUtils.resolve_unused_parameters!!(backend, gs, model)

        @test eltype(gs_resolved.w1) == Float32
        @test size(gs_resolved.w1) == (5, 3)
        @test eltype(gs_resolved.w2) == Float64
        @test size(gs_resolved.w2) == (2, 4)
        @test eltype(gs_resolved.w3) == Float32
        @test size(gs_resolved.w3) == (7,)
    end
end
