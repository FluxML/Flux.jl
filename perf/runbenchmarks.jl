using Flux
using InteractiveUtils # because versioninfo is defined there
versioninfo()
include("bench_utils.jl")

@info "Benchmark Dense"
include("dense.jl")

@info "Benchmark Conv"
include("conv.jl")

@info "Benchmark VGG"
include("vgg.jl")
