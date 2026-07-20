using Flux

versioninfo()
include("bench_utils.jl")

@info "Benchmark Dense"
include("dense.jl")

@info "Benchmark Conv"
include("conv.jl")

@info "Benchmark VGG"
include("vgg.jl")

@info "Benchmark Recurrent"
include("recurrent.jl")
