using Pkg

include(joinpath(@__DIR__, "test_utils_distributed.jl"))

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
# ENV["FLUX_TEST_REACTANT_CUDA"] = "true"

const FLUX_TEST_AMDGPU    = get(ENV, "FLUX_TEST_AMDGPU", "false") == "true"
const FLUX_TEST_CPU       = get(ENV, "FLUX_TEST_CPU",    "true")  == "true"
const FLUX_TEST_CUDA      = get(ENV, "FLUX_TEST_CUDA",   "false") == "true"
const FLUX_TEST_DIST_MPI  = get(ENV, "FLUX_TEST_DISTRIBUTED_MPI",  "false") == "true"
const FLUX_TEST_DIST_NCCL = get(ENV, "FLUX_TEST_DISTRIBUTED_NCCL", "false") == "true"
const FLUX_TEST_ENZYME    = get(ENV, "FLUX_TEST_ENZYME", "true") == "true"
const FLUX_TEST_METAL     = get(ENV, "FLUX_TEST_METAL",  "false") == "true"
const FLUX_TEST_REACTANT  = get(ENV, "FLUX_TEST_REACTANT", "true") == "true"
# Require the Reactant tests to run on XLA's CUDA backend instead of whatever Reactant selects.
const FLUX_TEST_REACTANT_CUDA = get(ENV, "FLUX_TEST_REACTANT_CUDA", "false") == "true"

# --- Optional package installation (main process, before workers start) ---
# Precompile once at the end: on Julia 1.13.2, `Pkg.add` fails to auto-precompile an
# indirect dependency whose version it changes (JuliaLang/Pkg.jl#4866).
withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
  FLUX_TEST_AMDGPU   && Pkg.add("AMDGPU")
  # (FLUX_TEST_ENZYME || FLUX_TEST_REACTANT) && Pkg.add("Enzyme") # Already installed
  FLUX_TEST_CUDA     && Pkg.add(["CUDA", "cuDNN"])
  FLUX_TEST_METAL    && Pkg.add("Metal")
  (FLUX_TEST_DIST_MPI || FLUX_TEST_DIST_NCCL) && Pkg.add("MPI")
  FLUX_TEST_DIST_NCCL && Pkg.add("NCCL")
  FLUX_TEST_REACTANT  && Pkg.add("Reactant")  # must come after CUDA
end
Pkg.precompile()

# Load only now: adding Reactant changes versions of some of Flux's dependencies, and loading
# Flux earlier would compile it a second time, against the versions that get replaced.
using Flux
using ParallelTestRunner

# --- Auto-discover all .jl files (except runtests.jl) ---
testsuite = find_tests(@__DIR__)

# The distributed child files are run only through the dedicated runner
# `ext_distributed/runtests.jl`; never discover them as individual tests.
strip_distributed_tests!(testsuite)

# --- Remove non-test utility files picked up by discovery ---
delete!(testsuite, "test_module")
delete!(testsuite, "test_utils_distributed")
delete!(testsuite, "ext_reactant/test_utils_reactant")
delete!(testsuite, "test_common/normalization")
delete!(testsuite, "test_common/gpu_recurrent")
delete!(testsuite, "test_common/gpu_train")

# --- Filter by env flags (remove disabled test groups) ---

## Uncomment and edit to run only a specific test file
# filter!(((k, _),) -> startswith(k, "layers/conv"), testsuite)

!FLUX_TEST_CPU      && filter!(((k, _),) -> startswith(k, "ext_"), testsuite)
!FLUX_TEST_CUDA     && filter!(((k, _),) -> !startswith(k, "ext_cuda"),        testsuite)
!FLUX_TEST_AMDGPU   && filter!(((k, _),) -> !startswith(k, "ext_amdgpu"),      testsuite)
!FLUX_TEST_METAL    && filter!(((k, _),) -> !startswith(k, "ext_metal"),        testsuite)
!FLUX_TEST_ENZYME   && filter!(((k, _),) -> !startswith(k, "ext_enzyme"),       testsuite)
!FLUX_TEST_REACTANT && filter!(((k, _),) -> !startswith(k, "ext_reactant"),     testsuite)

# Enable the dedicated distributed runner (its child files are never run
# individually) only when distributed tests are opted in.
if FLUX_TEST_DIST_MPI || FLUX_TEST_DIST_NCCL
    add_distributed_runner!(testsuite, joinpath(@__DIR__, "ext_distributed", "runtests.jl"))
end


# --- init_code: runs in every test subprocess before the test expression ---
# Constants are interpolated from the main process so subprocesses see the same values.
init_code = quote
    # Reactant/XLA preallocates 75% of the GPU by default, which starves the CUDA.jl tests that
    # share the card on CI (they OOM even loading a kernel — the tell is a ~100%-full GPU next to a
    # near-empty CUDA.jl pool). Disable preallocation and cap XLA's fraction so it allocates lazily
    # and both backends fit. These are Reactant's own knobs (the JAX-style XLA_PYTHON_CLIENT_* names
    # are ignored by Reactant); must be set before `using Reactant`.
    get!(ENV, "XLA_REACTANT_GPU_PREALLOCATE", "false")
    get!(ENV, "XLA_REACTANT_GPU_MEM_FRACTION", "0.25")
    # XLA's GPU conv fusion turns scalar reductions of a conv output into max-abs and fails to
    # compile some asymmetric paddings (EnzymeAD/Reactant.jl#3441). Also read before `using Reactant`.
    ENV["XLA_FLAGS"] = get(ENV, "XLA_FLAGS", "") * " --xla_gpu_experimental_enable_conv_fusion=false"
    const FLUX_TEST_ENZYME = $FLUX_TEST_ENZYME
    const FLUX_TEST_REACTANT_CUDA = $FLUX_TEST_REACTANT_CUDA
    using Random
    Random.seed!(0)
    include($(joinpath(@__DIR__, "test_module.jl")))
end

runtests(Flux, ARGS; testsuite, init_code)
