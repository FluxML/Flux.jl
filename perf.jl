using Flux, BenchmarkTools, Flux.Losses, CUDA
using Flux: OneHotMatrix

crossv2(ŷ::AbstractMatrix, y::AbstractVector{<:Integer}) = logitcrossentropy(ŷ, Flux.onehotbatch(y, 1:size(ŷ, 1)))
cross_unsafe(ŷ::AbstractMatrix, y::AbstractVector{<:Integer}) = logitcrossentropy(ŷ, OneHotMatrix(y, size(ŷ, 1)))

function perf(c, n)
    labels = rand(1:c, n)
    y = Flux.onehotbatch(labels, 1:c)
    ŷ = randn(Float32, c, n)
    
    labelsgpu = labels |> gpu
    ygpu = y |> gpu
    ŷgpu = ŷ |> gpu

    # println("with ŷ")
    # @btime logitcrossentropy($ŷ, $y); 
    # @btime gradient(ŷ -> logitcrossentropy(ŷ, $y), $ŷ);
    
    # println("with labels")
    # @btime logitcrossentropy($ŷ, $labels);
    # @btime gradient(ŷ -> logitcrossentropy(ŷ, $labels), $ŷ);

    # println("crossv2")
    # @btime crossv2($ŷ, $labels);
    # @btime gradient(ŷ -> crossv2(ŷ, $labels), $ŷ);
    
    # println("with ŷ - gpu")
    # @assert size(ŷgpu) == (c, n)
    # @btime CUDA.@sync logitcrossentropy($ŷgpu, $ygpu); 
    # @btime CUDA.@sync gradient(ŷ -> logitcrossentropy(ŷ, $ygpu), $ŷgpu);
    
    # println("with labels - gpu")
    # @btime CUDA.@sync logitcrossentropy($ŷgpu, $labelsgpu);
    # @btime CUDA.@sync gradient(ŷ -> logitcrossentropy(ŷ, $labelsgpu), $ŷgpu);

    # println("crossv2 - gpu")
    # @btime CUDA.@sync crossv2($ŷgpu, $labelsgpu);
    # @btime CUDA.@sync gradient(ŷ -> crossv2(ŷ, $labelsgpu), $ŷgpu);

    println("cross_unsafe - gpu")
    @btime CUDA.@sync cross_unsafe($ŷgpu, $labelsgpu);
    @btime CUDA.@sync gradient(ŷ -> cross_unsafe(ŷ, $labelsgpu), $ŷgpu);
    
    return nothing
end

perf(10, 128)
# with ŷ
#   14.648 μs (10 allocations: 13.17 KiB)
#   27.381 μs (19 allocations: 35.39 KiB)
# with labels
#   13.716 μs (16 allocations: 9.88 KiB)
#   41.338 μs (119 allocations: 25.22 KiB)
# crossv2
#   14.838 μs (11 allocations: 13.73 KiB)
#   27.501 μs (20 allocations: 35.95 KiB)
# with ŷ - gpu
#   46.107 μs (163 allocations: 8.52 KiB)
#   109.656 μs (414 allocations: 24.17 KiB)
# with labels - gpu
#   42.620 μs (125 allocations: 6.23 KiB)
#   117.972 μs (375 allocations: 19.61 KiB)
# crossv2 - gpu
#   107.913 μs (284 allocations: 14.45 KiB)
#   177.093 μs (535 allocations: 30.11 KiB)
# cross_unsafe - gpu
#   46.647 μs (163 allocations: 8.52 KiB)
#   110.759 μs (414 allocations: 24.17 KiB)

perf(100, 128)
# with ŷ
#   121.148 μs (12 allocations: 103.02 KiB)
#   212.059 μs (25 allocations: 304.92 KiB)
# with labels
#   113.914 μs (17 allocations: 54.80 KiB)
#   215.665 μs (122 allocations: 159.98 KiB)
# crossv2
#   122.620 μs (13 allocations: 103.58 KiB)
#   215.615 μs (26 allocations: 305.48 KiB)
# with ŷ - gpu
#   47.880 μs (163 allocations: 8.52 KiB)
#   110.307 μs (414 allocations: 24.17 KiB)
# with labels - gpu
#   40.567 μs (125 allocations: 6.23 KiB)
#   122.961 μs (375 allocations: 19.61 KiB)
# crossv2 - gpu
#   104.917 μs (284 allocations: 14.45 KiB)
#   171.141 μs (535 allocations: 30.11 KiB)
# cross_unsafe - gpu
#   46.137 μs (163 allocations: 8.52 KiB)
#   109.084 μs (414 allocations: 24.17 KiB)

perf(100, 1280)
# with ŷ
#   1.378 ms (12 allocations: 1.00 MiB)
#   2.320 ms (25 allocations: 2.97 MiB)
# with labels
#   1.321 ms (18 allocations: 540.97 KiB)
#   2.169 ms (123 allocations: 1.52 MiB)
# crossv2
#   1.384 ms (13 allocations: 1.01 MiB)
#   2.317 ms (26 allocations: 2.98 MiB)
# with ŷ - gpu
#   60.885 μs (210 allocations: 10.77 KiB)
#   121.919 μs (464 allocations: 26.47 KiB)
# with labels - gpu
#   52.679 μs (174 allocations: 8.52 KiB)
#   128.602 μs (426 allocations: 21.92 KiB)
# crossv2 - gpu
#   137.448 μs (422 allocations: 20.91 KiB)
#   208.361 μs (676 allocations: 36.61 KiB)
# cross_unsafe - gpu
#   58.479 μs (210 allocations: 10.77 KiB)
#   121.839 μs (464 allocations: 26.47 KiB)
