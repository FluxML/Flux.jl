using Pkg
Pkg.activate(".")

using Flux, BenchmarkTools, LinearAlgebra
using CUDA

cu_x = rand(CURAND.default_rng(), Float32, 100, 100)
x = rand(Float32, 100, 100)
y = Flux.onehotbatch(1:100, 1:100)

@which x*y #*(A::AbstractArray{T,2} where T, B::Flux.OneHotMatrix) at Flux.jl\src\onehot.jl:40
@which x*y' #*(A::AbstractArray{T,2} where T, B::Adjoint{Bool,var"#s126"} where var"#s126"<:Flux.OneHotMatrix) at Flux.jl\src\onehot.jl:52
@btime x*y
# 2.663 μs (2 allocations: 39.14 KiB)
@btime x*y'
# 3.686 μs (3 allocations: 39.17 KiB)

@which cu_x*y #*(A::CuArray, B::Flux.OneHotMatrix) at Flux.jl\src\onehot.jl:37
@which cu_x*y' #*(A::CuArray, B::Adjoint{Bool,var"#s126"} where var"#s126"<:Flux.OneHotMatrix) at Flux.jl\src\onehot.jl:38

CUDA.allowscalar(false)
@btime CUDA.@sync cu_x*y
# 70.400 μs (75 allocations: 2.83 KiB)
@btime CUDA.@sync cu_x*y'
# 93.699 μs (21 allocations: 78.92 KiB)
