using Test
using Random
using MLUtils

include(joinpath(@__DIR__, "distributed_setup.jl"))

rng = Xoshiro(1234)

# Expected cyclic-padding spec for N observations across W workers:
#   per   = cld(N, W)          (every rank receives `per` observations)
#   total = per * W            (padded length of the global index list)
#   full  = [mod1(i, N) for i in 1:total]
# Rank `r` (0-based) receives full[(r*per + 1):((r+1)*per)].
function _cyclic_spec(N::Int, W::Int)
    per = cld(N, W)
    total = per * W
    full = [mod1(i, N) for i in 1:total]
    return per, total, full
end

@testset "DistributedDataContainer cyclic padding" begin
    for N in (1, 2, 7, 10)
        @testset "N = $N" begin
            per, total, full = _cyclic_spec(N, tworkers)
            @test total == per * tworkers
            @test length(full) == total
            @test all(i -> 1 <= i <= N, full)
            @test full[1:N] == collect(1:N)
            @test full[(N + 1):end] == [mod1(j, N) for j in 1:(total - N)]

            chunk = full[(rank * per + 1):((rank + 1) * per)]
            @test length(chunk) == per
            @test all(i -> 1 <= i <= N, chunk)

            @testset "identity index container" begin
                data_id = Float32.(1:N)
                dc_id = DistributedUtils.DistributedDataContainer(backend, data_id)
                @test length(dc_id) == cld(N, tworkers)
                @test [MLUtils.getobs(dc_id, k) for k in 1:length(dc_id)] == chunk
                @test all(v -> 1 <= v <= N,
                    [MLUtils.getobs(dc_id, k) for k in 1:length(dc_id)])
            end

            @testset "duplicate aggregate via allreduce" begin
                data_rand = randn(rng, Float32, N)
                dc_r = DistributedUtils.DistributedDataContainer(backend, data_rand)
                local_sum = 0.0f0
                local_ok = true
                try
                    local_sum = sum(MLUtils.getobs(dc_r, k) for k in 1:length(dc_r))
                catch e
                    local_ok = false
                    @error "unexpected error while aggregating the local shard" exception = (e, catch_backtrace())
                end
                # Every rank must reach the collective even if a rank failed
                # locally, so that a single bad rank cannot deadlock the others.
                global_sum = DistributedUtils.allreduce!(backend, [local_sum], +)[1]
                @test local_ok
                @test global_sum ≈ sum(data_rand[full])
            end
        end
    end

    @testset "N = 0 empty dataset rejected" begin
        err = try
            DistributedUtils.DistributedDataContainer(backend, Float32[])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test err !== nothing && occursin("empty", sprint(showerror, err))
        @test err !== nothing && occursin("numobs(data) == 0", sprint(showerror, err))
    end
end
