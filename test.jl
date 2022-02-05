using Flux, CUDA, LinearAlgebra, BenchmarkTools, NNlib, NNlibCUDA
using Flux: onehotbatch

function mul0(A::AbstractMatrix, B::Flux.OneHotMatrix)
  A * B
end

function mul0(A::AbstractMatrix, B::Adjoint{Bool,<:Flux.OneHotMatrix})
  A * B
end

function mul1(A::AbstractMatrix, B::Flux.OneHotMatrix)
  m = size(A,1)
  Y = similar(A, m, size(B,2))
  for (j, ix) in enumerate(B.indices)
    for i in 1:m
      @inbounds Y[i,j] = A[i,ix]
    end
  end
  Y
end

function mul1(A::AbstractMatrix, B::Adjoint{Bool,<:Flux.OneHotMatrix})
  m = size(A,1)
  Y = fill!(similar(A, m, size(B,2)), zero(eltype(A)))
  for (j, ix) in enumerate(parent(B).indices)
      for i in 1:m
          @inbounds Y[i,ix] += A[i,j]
      end
  end
  Y
end

function mul2(A::AbstractMatrix, B::Flux.OneHotMatrix)
  NNlib.gather(A, B.indices)
end

function mul2(A::AbstractMatrix, B::Adjoint{Bool,<:Flux.OneHotMatrix})
  NNlib.scatter(+, A, parent(B).indices, dstsize=(size(A,1), size(B,2)))
end

bs = 128;
Din = 100;
Dout = Din;

A = rand(Float32, Dout, Din);
oh = onehotbatch(rand(1:Din, bs), 1:Din);

@assert mul0(A,oh) == mul1(A,oh) == mul2(A,oh)

# println("# mul0")
# @btime mul0($A, $oh);
# println("# mul1")
# @btime mul1($A, $oh);
# println("# mul2")
# @btime mul2($A, $oh);

# gA, goh = A |> gpu, oh |> gpu;

# println("# gpu mul0")
# @btime mul0($gA, $goh);
# println("# gpu mul1")
# @btime mul1($gA, $goh);
# println("# gpu mul2")
# @btime mul2($gA, $goh);


# grad0 = gradient(A -> sum(mul0(A, oh)), A)[1]
# gradg0 = gradient(A -> sum(mul0(A, goh)), gA)[1]
# @assert Array(gradg0) ≈ grad0

# grad2 = gradient(A -> sum(mul2(A, oh)), A)[1]
# gradg2 = gradient(A -> sum(mul2(A, goh)), gA)[1]
# @assert grad2 ≈ grad0
# @assert Array(gradg2) ≈ grad2

# println("# grad mul0")
# @btime gradient(A -> sum(mul0(A, $oh)), $A)[1]
# # println("# grad mul1") # errors out since mutates
# # @btime gradient(A -> sum(mul1(A, oh)), A)[1]
# println("# grad mul2")
# @btime gradient(A -> sum(mul2(A, $oh)), $A)[1]

# println("# grad gpu mul0")
# @btime gradient(A -> sum(mul0(A, $goh)), $gA)[1]
# # println("# grad mul1") # errors out since mutates
# # @btime gradient(A -> sum(mul1(A, oh)), A)[1]
# println("# grad gpu mul2")
# @btime gradient(A -> sum(mul2(A, $goh)), $gA)[1]

B = rand(Float32, Dout, bs);

@assert mul1(B, oh') ≈ mul0(B, oh')
@assert mul2(B, oh') ≈ mul0(B, oh')

println("# adjoint mul0")
@btime mul0($B, $(oh'));
println("# adjoint mul1")
@btime mul1($B, $(oh'));
println("# adjoint mul2")
@btime mul2($B, $(oh'));


