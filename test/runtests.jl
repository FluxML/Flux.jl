# Pkg.test runs with --check_bounds=1, forcing all bounds checks.
# This is incompatible with CUDAnative (see JuliaGPU/CUDAnative.jl#98)
if Base.JLOptions().check_bounds == 1
  file = @__FILE__
  run(```
    $(Base.julia_cmd())
    --color=$(Base.have_color ? "yes" : "no")
    --compiled-modules=$(Bool(Base.JLOptions().use_compiled_modules) ? "yes" : "no")
    --startup-file=$(Base.JLOptions().startupfile != 2 ? "yes" : "no")
    --code-coverage=$(["none", "user", "all"][1+Base.JLOptions().code_coverage])
    $(file)
    ```)
  exit()
end

using Flux, Test, Random
using Random

Random.seed!(0)

# So we can use the system CuArrays
insert!(LOAD_PATH, 2, "@v#.#")

@testset "Flux" begin

include("utils.jl")
include("onehot.jl")
include("tracker.jl")
include("layers/normalisation.jl")
include("layers/stateless.jl")
include("optimise.jl")
include("data.jl")

if Base.find_package("CuArrays") != nothing
  include("cuda/cuda.jl")
end

end
