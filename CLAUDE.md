# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all CPU tests:**
```
julia --project -e 'using Pkg; Pkg.test()'
```

**Run a single test file:**
```
julia --project test/losses.jl
julia --project test/layers/basic.jl
```

**Run tests with specific backends:**
```
FLUX_TEST_CUDA=true julia --project -e 'using Pkg; Pkg.test()'
FLUX_TEST_ENZYME=false julia --project -e 'using Pkg; Pkg.test()'
FLUX_TEST_REACTANT=false julia --project -e 'using Pkg; Pkg.test()'
```

Test environment flags: `FLUX_TEST_CPU` (default true), `FLUX_TEST_CUDA`, `FLUX_TEST_AMDGPU`, `FLUX_TEST_METAL`, `FLUX_TEST_ENZYME` (default true on Julia < 1.12), `FLUX_TEST_REACTANT` (default true), `FLUX_TEST_DISTRIBUTED_MPI`, `FLUX_TEST_DISTRIBUTED_NCCL`.

**Format code:**
```
julia -e 'using JuliaFormatter; format("src")'
```
Config: indent=4, margin=80, `always_for_in=true` (see `.JuliaFormatter.toml`).

There is a pre-commit hook in `.githooks/pre-commit` that auto-formats staged files. Activate it with:
```
git config core.hooksPath .githooks
```

## Architecture

Flux.jl is a pure-Julia ML framework. The main entry point is [src/Flux.jl](src/Flux.jl), which re-exports from several key dependencies.

### Core Design

**Gradient computation** ([src/gradient.jl](src/gradient.jl)): `Flux.gradient` and `Flux.withgradient` are thin wrappers that default to Zygote, but dispatch to Enzyme when `Duplicated` arguments are detected. Other backends (Mooncake, FiniteDifferences) are specified via `ADTypes` (`AutoMooncake()` etc.) and loaded via package extensions.

**Training API** ([src/train.jl](src/train.jl)): `Flux.setup(rule, model)` wraps `Optimisers.setup` with mutability checks. `Flux.train!(loss, model, data, opt_state)` iterates over data calling gradients and `Optimisers.update!`. Models are mutated in-place.

**Layer definition** ([src/layers/macro.jl](src/layers/macro.jl)): The `@layer` macro is the standard way to declare a custom layer. It calls `Functors.@functor` (so `fmap` traverses parameters) and sets up pretty-printing.

**Functor integration** ([src/functor.jl](src/functor.jl)): `fmap` from Functors.jl is used to move models between devices (`gpu`, `cpu`) and change precision (`f32`, `f64`, `f16`). The `trainable` function (from Optimisers.jl) controls which parameters are updated.

### Key Dependencies (re-exported to users)

- **NNlib** — low-level primitives: `conv`, `relu`, `softmax`, pooling, etc.
- **Optimisers.jl** — optimizer rules (`Adam`, `SGD`, etc.) and `destructure`
- **Zygote.jl** — default AD engine; `pullback`, `jacobian`, `hessian` re-exported
- **MLDataDevices.jl** — device abstraction (`gpu_device()`, `cpu_device()`, `CUDADevice`, etc.)
- **ADTypes.jl** — `AutoZygote`, `AutoEnzyme`, `AutoMooncake`, `AutoFiniteDifferences`
- **Functors.jl** — `fmap`, `fmapstructure` for model traversal
- **OneHotArrays.jl** — `onehot`, `onehotbatch`, `onecold`

### Source Layout

```
src/
├── gradient.jl       # Flux.gradient / withgradient with multi-backend dispatch
├── train.jl          # Flux.setup, Flux.train! (Train submodule)
├── functor.jl        # fmap-based device movement, @functor helpers
├── utils.jl          # weight initializers (glorot_uniform, kaiming_normal, …), nfan
├── outputsize.jl     # @autosize macro, layer output shape inference
├── loading.jl        # model state loading/saving
├── devices.jl        # Flux.get_device (retrocompat wrapper)
├── layers/
│   ├── macro.jl      # @layer macro
│   ├── basic.jl      # Dense, Chain, Parallel, SkipConnection, Embedding, …
│   ├── conv.jl       # Conv, ConvTranspose, DepthwiseConv, pooling layers
│   ├── recurrent.jl  # RNNCell/RNN, LSTMCell/LSTM, GRUCell/GRU, Recurrence
│   ├── normalise.jl  # BatchNorm, LayerNorm, InstanceNorm, GroupNorm, WeightNorm
│   ├── attention.jl  # MultiHeadAttention
│   └── upsample.jl   # Upsample, PixelShuffle
├── losses/
│   └── functions.jl  # mse, crossentropy, focal_loss, kldivergence, …
└── distributed/      # MPIBackend, NCCLBackend, DistributedUtils
```

### Extension System

Optional backends live in `ext/` as Julia package extensions (weak dependencies):

| Extension | Trigger package | Purpose |
|---|---|---|
| `FluxCUDAExt` | CUDA.jl | NVIDIA GPU support |
| `FluxAMDGPUExt` | AMDGPU.jl | AMD GPU support |
| `FluxEnzymeExt` | Enzyme.jl | Enzyme AD backend |
| `FluxMooncakeExt` | Mooncake.jl | Mooncake AD backend |
| `FluxMPIExt` | MPI.jl | Distributed training |
| `FluxMPINCCLExt` | NCCL.jl | NCCL all-reduce |

### Test Layout

Tests mirror the source structure. [test/test_utils.jl](test/test_utils.jl) provides `test_gradients`, which checks a layer's gradient against multiple AD backends. [test/testsuite/normalization.jl](test/testsuite/normalization.jl) is a reusable test suite run for each device backend.

## GitHub Repository

**Repo:** https://github.com/FluxML/Flux.jl (under the `FluxML` org)

**CI systems:**
- **GitHub Actions** (`.github/workflows/ci.yml`): CPU tests on Julia 1.10 (minimum), latest stable, and nightly; platforms: Ubuntu x64, Windows x64, macOS aarch64.
- **Buildkite** (`.buildkite/pipeline.yml`): GPU tests on JuliaGPU infrastructure — CUDA (Julia 1), Metal (macOS aarch64), AMDGPU (ROCm). These run with `FLUX_TEST_CPU=false` and the relevant GPU flag set to `true`.

**PR checklist** (from open PRs convention): tests added, entry in `NEWS.md`, documentation updated if applicable.
