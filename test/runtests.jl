using Pkg
using Flux
using ParallelTestRunner

# --- Env flags ---

## Uncomment below to change the default test settings
# ENV["FLUX_TEST_AMDGPU"] = "true"
# ENV["FLUX_TEST_CUDA"] = "true"
# ENV["FLUX_TEST_METAL"] = "true"
# ENV["FLUX_TEST_CPU"] = "false"
# ENV["FLUX_TEST_DISTRIBUTED_MPI"] = "true"
# ENV["FLUX_TEST_DISTRIBUTED_NCCL"] = "true"
# ENV["FLUX_TEST_ENZYME"] = "false"
# ENV["FLUX_TEST_REACTANT"] = "true"

const FLUX_TEST_AMDGPU    = get(ENV, "FLUX_TEST_AMDGPU", "false") == "true"
const FLUX_TEST_CPU       = get(ENV, "FLUX_TEST_CPU",    "true")  == "true"
const FLUX_TEST_CUDA      = get(ENV, "FLUX_TEST_CUDA",   "false") == "true"
const FLUX_TEST_DIST_MPI  = get(ENV, "FLUX_TEST_DISTRIBUTED_MPI",  "false") == "true"
const FLUX_TEST_DIST_NCCL = get(ENV, "FLUX_TEST_DISTRIBUTED_NCCL", "false") == "true"
const FLUX_TEST_ENZYME    = get(ENV, "FLUX_TEST_ENZYME", VERSION < v"1.12-" ? "true" : "false") == "true"
const FLUX_TEST_METAL     = get(ENV, "FLUX_TEST_METAL",  "false") == "true"
const FLUX_TEST_REACTANT  = get(ENV, "FLUX_TEST_REACTANT", "true") == "true"

# --- Optional package installation (main process, before workers start) ---
FLUX_TEST_AMDGPU   && Pkg.add("AMDGPU")
(FLUX_TEST_ENZYME || FLUX_TEST_REACTANT) && Pkg.add("Enzyme")
FLUX_TEST_CUDA     && Pkg.add(["CUDA", "cuDNN"])
FLUX_TEST_METAL    && Pkg.add("Metal")
(FLUX_TEST_DIST_MPI || FLUX_TEST_DIST_NCCL) && Pkg.add("MPI")
FLUX_TEST_DIST_NCCL && Pkg.add("NCCL")
FLUX_TEST_REACTANT  && Pkg.add("Reactant")  # must come after CUDA

# --- Auto-discover all .jl files (except runtests.jl) ---
testsuite = find_tests(@__DIR__)

# --- Remove non-test utility files picked up by discovery ---
delete!(testsuite, "test_module")
delete!(testsuite, "ext_reactant/test_utils_reactant")
delete!(testsuite, "testsuite/normalization")

# --- Filter by env flags (remove disabled test groups) ---

## Uncomment and edit to run only a specific test file
# filter!(((k, _),) -> startswith(k, "layers/conv"), testsuite)

!FLUX_TEST_CPU      && filter!(((k, _),) -> startswith(k, "ext_"), testsuite)
!FLUX_TEST_CUDA     && filter!(((k, _),) -> !startswith(k, "ext_cuda"),        testsuite)
!FLUX_TEST_AMDGPU   && filter!(((k, _),) -> !startswith(k, "ext_amdgpu"),      testsuite)
!FLUX_TEST_METAL    && filter!(((k, _),) -> !startswith(k, "ext_metal"),        testsuite)
!(FLUX_TEST_DIST_MPI || FLUX_TEST_DIST_NCCL) &&
    filter!(((k, _),) -> !startswith(k, "ext_distributed"), testsuite)
!FLUX_TEST_ENZYME   && filter!(((k, _),) -> !startswith(k, "ext_enzyme"),       testsuite)
!FLUX_TEST_REACTANT && filter!(((k, _),) -> !startswith(k, "ext_reactant"),     testsuite)


# --- init_code: runs in every test subprocess before the test expression ---
# Constants are interpolated from the main process so subprocesses see the same values.
init_code = quote
    const FLUX_TEST_ENZYME = $FLUX_TEST_ENZYME
    using Random
    Random.seed!(0)
    include($(joinpath(@__DIR__, "test_module.jl")))
    include($(joinpath(@__DIR__, "testsuite", "normalization.jl")))  # defines normalization_testsuite
end

runtests(Flux, ARGS; testsuite, init_code)
