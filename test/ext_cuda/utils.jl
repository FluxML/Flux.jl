

@testset "Rrule" begin
  @testset "issue 2033" begin
    struct Wrapped{T}
        x::T
    end
    y, _ = Flux.pullback(Wrapped, cu(randn(3,3)))
    @test y isa Wrapped{<:CuArray}
  end
end

@testset "rng_from_array" begin
    x = cu(randn(2,2))
    rng = Flux.rng_from_array(x)
    @test rng == CUDA.default_rng()
end
