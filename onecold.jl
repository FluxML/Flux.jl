using CUDA
using Flux
using Flux: onehotbatch, onecold, OneHotMatrix
using BenchmarkTools
CUDA.allowscalar(false)

onecold2(y::AbstractMatrix, labels=1:size(y,1)) =
  vec(map(x -> labels[x[1]], argmax(y; dims=1)))

onecold2(y::OneHotMatrix, labels...) = 
    map(x -> Flux.onecold(x, labels...), y.data)

function accuracy_v1a(oh, ŷ)
    mean(onecold(oh) .== onecold(ŷ))
end

function accuracy_v1b(oh, ŷ)
    mean(onecold(cpu(oh)) .== onecold(cpu(ŷ)))
end

function accuracy_v1c(y, ŷ)
    mean(cpu(y) .== onecold(cpu(ŷ)))
end

function accuracy_v2a(oh, ŷ)
    mean(onecold2(oh) .== onecold2(ŷ))
end

function accuracy_v2b(oh, ŷ)
    mean(onecold2(cpu(oh)) .== onecold2(cpu(ŷ)))
end

function accuracy_v2c(y, ŷ)
    mean(y .== onecold2(ŷ))
end

function accuracy_v3(y, ŷ)
    mean(y .== mapslices(argmax, ŷ, dims=1))
end

function accuracy_v4(oh, ŷ)
    mean(maximum(oh .* ŷ, dims=1) .== maximum(ŷ, dims=1))
end


ŷ = rand(Float32, 100, 1000)
y = rand(1:100, 1000)
oh = onehotbatch(y, 1:100)
ŷg, yg, ohg = gpu.([ŷ, y, oh]) 

println("V1A")
@btime accuracy_v1a(oh, ŷ)
# @btime CUDA.@sync accuracy_v1a(ohg, ŷg) # Error scalar indexing

println("\nV1B")
@btime accuracy_v1b(oh, ŷ)
@btime CUDA.@sync accuracy_v1b(ohg, ŷg)

println("\nV1C")
@btime accuracy_v1c(y, ŷ)
@btime CUDA.@sync accuracy_v1c(yg, ŷg)

println("\nV2A")
@btime accuracy_v2a(oh, ŷ)
@btime CUDA.@sync accuracy_v2a(ohg, ŷg)

println("\nV2B")
@btime accuracy_v2b(oh, ŷ)
@btime CUDA.@sync accuracy_v2b(ohg, ŷg)

println("\nV2C")
@btime accuracy_v2c(y, ŷ)
@btime CUDA.@sync accuracy_v2c(yg, ŷg)

println("\nV3")
@btime accuracy_v3(y, ŷ)
# @btime CUDA.@sync accuracy_v3(yg, ŷg) # Error scalar indexing

println("\nV4")
@btime accuracy_v4(oh, ŷ)
@btime CUDA.@sync accuracy_v4(ohg, ŷg) 