# Flux Release Notes

See also [github's page](https://github.com/FluxML/Flux.jl/releases) for a complete list of PRs merged before each release.

## Unreleased

- Add `lux-reactant-nitro/` versions of the GPT-2/FineWeb-Edu and ResNet-18/Tiny-ImageNet examples, which train Lux models with [ReactantNitro.jl](https://github.com/EnzymeAD/ReactantNitro.jl) for comparison with the Flux versions, and share their data pipelines with them ([#2803](https://github.com/FluxML/Flux.jl/pull/2803)).
- Refresh the GPT-2/FineWeb-Edu example and benchmarks for current Flux and NNlib master, which keeps Reactant activations in bfloat16 with Float32 LayerNorm parameters. Disabling evaluation also skips compiling its loss, and the PyTorch port requires `datasets` 5.1 for correct training and validation splits ([#2794](https://github.com/FluxML/Flux.jl/pull/2794)).
- Refresh the ResNet-18/Tiny-ImageNet example with separate CUDA.jl and Reactant environments, integer class labels, automatic training GC, and a matching PyTorch port. The CUDA version uses CUDA/cuDNN 6.5, and MLDataDevices 1.18 frees each previous batch on Reactant too ([#2794](https://github.com/FluxML/Flux.jl/pull/2794)).
- `trainstep!` and `trainstep_withgradient!` run garbage collections between steps, as `train!` does, so a custom training loop on a GPU no longer needs `GC.gc` calls to keep memory in check. On the CPU, they make compute-bound models train faster. The collections are paced for each model, across calls of `trainstep!` and `train!`. The new keyword `gc_interval` sets how often, with the same values as for `train!`; `gc_interval = 0` turns them off. Nothing changes on a Reactant device ([#2795](https://github.com/FluxML/Flux.jl/issues/2795)).
- `train!` runs a full garbage collection after a step that compiles, that is, the first one with the types of its arguments (loss, model, batch and optimiser state). On a GPU, the arrays of that step could otherwise stay allocated for the rest of training, since the incremental collections that `train!` runs don't free them. For a ResNet-18 with batch size 128 on CUDA, that was 3 GiB, which raised the peak memory in use from 8.4 to 11.2 GiB ([#2795](https://github.com/FluxML/Flux.jl/issues/2795)).
- Fix `WeightNorm(layer; dims = :)`, whose magnitude `g` was a scalar instead of an array, so that optimisers silently never trained it. It is now a `1×1` array for a matrix weight, normalising over all of the weight's entries as before ([#2796](https://github.com/FluxML/Flux.jl/issues/2796)).
- The Mooncake rule for `Flux.Losses.mse` moves here from Mooncake's own Flux extension. It is loaded from Mooncake 0.5.67, which no longer ships that extension, so nothing changes with earlier Mooncake releases ([#2772](https://github.com/FluxML/Flux.jl/pull/2772), [chalk-lab/Mooncake.jl#1371](https://github.com/chalk-lab/Mooncake.jl/issues/1371)).
- `early_stopping` and `plateau` count a change of exactly `min_dist` as no improvement, as their docstrings say. With the default `min_dist = 0`, `early_stopping` used to never trigger on a loss that stays constant, as each repeated value reset its counter ([#2791](https://github.com/FluxML/Flux.jl/issues/2791)).
- Lower time to first gradient: Flux has a precompile workload, which runs the forward pass, `gradient`, `withgradient`, `setup`, `update!` and `train!` on small `Float32` models made of `Dense`, `Conv`, `BatchNorm`, `Dropout` and pooling layers. In a fresh session, loading Flux, building a small MLP or CNN and taking its first training step takes about 2.5 s instead of 8–16 s. Models with other layers also gain: with `LayerNorm` it takes 5 s instead of 11 s, and with `LSTM` 8 s instead of 14 s. Precompiling Flux takes about 16 s longer. Flux now requires Zygote 0.7.15, which fixes wrong gradients through precompiled code on Julia 1.12 and later ([#2781](https://github.com/FluxML/Flux.jl/issues/2781)).
- Fix `Flux.withgradient`, `Flux.trainstep!` and `Flux.train!` with Enzyme for a loss returning a `NamedTuple` with numeric auxiliary outputs, such as `(loss=mse(m(x), y), pred=m(x))`: since v0.16.14, the gradient could silently be zero, depending on the model ([#2790](https://github.com/FluxML/Flux.jl/pull/2790)).
- Fix the gradient of `logitcrossentropy` for a single sample with a 0-dimensional class label, such as `logitcrossentropy(ŷ, fill(2))` for a vector `ŷ`, which threw a `MethodError` ([#2787](https://github.com/FluxML/Flux.jl/issues/2787)).

## v0.16.14 (24 October 2026)

- `logitcrossentropy` with class labels (integers or a `OneHotArray`) computes its gradient in one pass over the logits, in their element type, and no longer copies `Float16`/`BFloat16` logits to `Float32`, as `NNlib.logsumexp` now accumulates them in `Float32` itself. For GPT-2's `BFloat16` logits on CUDA (50304 × 8192), the loss takes 4 ms instead of 6 ms, and its gradient 6 ms and 2.3 GiB instead of 14 ms and 8.4 GiB. On the CPU, the gradient is 1.1–3× faster ([#2767](https://github.com/FluxML/Flux.jl/issues/2767)).
- A layer can customize how `f16`, `bf16`, `f32` and `f64` convert it, wherever it appears in a model: these functions call a 2-argument method, such as `f16(layer, cx)`, on every part of the model. For example, `Flux.bf16(m::MyNorm, cx) = f32(m, cx)` keeps a custom normalization layer in `Float32`, as `BatchNorm`, `InstanceNorm`, `GroupNorm` and `LayerNorm` are. Tied arrays stay shared across such methods ([#2782](https://github.com/FluxML/Flux.jl/issues/2782)).
- Differentiating with Zygote through `f16`, `bf16`, `f32` or `f64`, as in `gradient(m -> loss(bf16(m), x), m)`, no longer drops the gradient of every use but the first of an array that appears several times in the model, such as tied weights. It is also much faster: the conversion of a 500-node model used to add 6 ms to a Zygote gradient, and now adds 0.14 ms ([#2784](https://github.com/FluxML/Flux.jl/pull/2784)).
- Lower first-call latency: the first construction of a `Conv`, `CrossCor` or `DepthwiseConv` layer in a session takes about 0.4 s instead of 1.4 s, as its `pad_mode` check no longer compiles `repr` for an error message it does not throw. The size checks in the forward pass of `Dense`, normalization and recurrent layers, and in `WeightNorm` and `loadmodel!`, also build their error messages lazily ([#2780](https://github.com/FluxML/Flux.jl/issues/2780)).
- `focal_loss` and `tversky_loss` no longer accept the keywords `γ`, `ϵ` and `β`, which were silently ignored since the deprecation of the Greek keyword names was removed. Pass `gamma`, `eps` and `beta` instead ([#2778](https://github.com/FluxML/Flux.jl/issues/2778)).
- `LayerNorm` computes its forward pass with `NNlib.layernorm`, instead of `NNlib.normalise` followed by the `Scale` layer. It is faster, uses cuDNN for `Float32` and `Float64` CUDA arrays, and normalises `Float16`/`BFloat16` input in `Float32`. Like the other normalization layers, `LayerNorm` is now converted by `f16` and `bf16` in mixed precision, keeping its scale and bias in `Float32`, and its output has the element type of the input. Note the behavior change: `LayerNorm` parameters are no longer downcast to `Float16`/`BFloat16` ([#2771](https://github.com/FluxML/Flux.jl/issues/2771)).
- Layers with `BFloat16` weights no longer copy `BFloat16` input, and accept one-hot input, which threw a method ambiguity error ([#2766](https://github.com/FluxML/Flux.jl/issues/2766)).
- Fix the gradient of `Flux.Losses.xlogx`, whose rule threw an `UndefVarError` with Zygote outside of broadcasting, e.g. in `sum(xlogx, x)`, and with any AD that uses ChainRules scalar rules ([#2762](https://github.com/FluxML/Flux.jl/issues/2762)).
- `crossentropy` and `logitcrossentropy` accept integer class labels as `y`, as PyTorch's `cross_entropy` does: an array with the size of `ŷ` without dimension `dims`, holding labels in `1:size(ŷ, dims)`, e.g. a vector of length `size(ŷ, 2)` for a matrix `ŷ`. The loss picks the entries of `ŷ` of the true classes instead of building a one-hot array of the size of `ŷ`, which saves memory and time for many classes, as in language models. With `agg = identity`, the losses have the size of `y`. A `y::OneHotArray` now takes the same path, with the same results as before ([#2756](https://github.com/FluxML/Flux.jl/issues/2756)).
- All losses compute in `Float32` when given `Float16` or `BFloat16` inputs, and return a `Float32`. They used to compute in the input's type, whose sums are inaccurate over many elements: `logitcrossentropy` on `BFloat16` logits of a language model batch could return half the correct value, and on `Float16` logits `Inf`. The gradient with respect to `ŷ` keeps its element type, so a model converted with `bf16` or `f16` still gets low-precision gradients, on the CPU, on GPUs, and under Reactant ([#2757](https://github.com/FluxML/Flux.jl/issues/2757)).

## v0.16.13 (19 October 2026)

### Distributed training fixes

- **PMI guardrail**: Initializing a distributed backend now checks the launch
  environment before `MPI.Init()` (including the `MPI.Init()` performed while
  bootstrapping `NCCLBackend`) and errors when the job was started by a
  PMIx/OpenMPI launcher (i.e. `PMIX_RANK` or `OMPI_COMM_WORLD_RANK` is set)
  while MPICH is the loaded MPI library — a known unsafe combination that
  previously caused a native hard abort. System OpenMPI launched with PMIx
  passes the check. Pass `force=true` to the same `initialize(...)` call (for
  either `MPIBackend` or `NCCLBackend`) to bypass the check (expert use only)
  ([#2694](https://github.com/FluxML/Flux.jl/pull/2694)).
- **Data sharding**: `DistributedDataContainer` now gives every worker an
  equal-length shard. When the number of observations `N` is not divisible by the
  number of workers, it pads its index sequence by repeating observations
  cyclically, so each worker receives `cld(N, workers)` observations while every
  index stays in `1:N`. The underlying dataset is not modified. Duplicated
  observations are included in training batches, so they receive extra weight in
  the averaged gradients and change the effective training objective. They also
  affect epoch accounting and metrics computed over the sharded data. An empty
  dataset (`numobs(data) == 0`) is rejected with an `ArgumentError`
  ([#2694](https://github.com/FluxML/Flux.jl/pull/2694)).

- New `Attention` layer, replacing `MultiHeadAttention` which is now deprecated. `Attention` returns only the transformed sequence, with the attention weights available through `Flux.attention_with_weights`, which returns both. This lets it use the fused attention of NNlib 0.9.48, which does not materialize the attention weights, so that it is faster and its memory grows linearly instead of quadratically with the sequence length, on the CPU and on GPUs, and with cuDNN's kernel for a layer converted with `f16` or `bf16` on CUDA. The new layer also has an `is_causal` option, padding given by sequence lengths (`q_lengths`, `kv_lengths`), grouped-query attention (`nkvheads`), normalization of queries and keys (`qk_norm`), and accepts inputs with any number of batch dimensions. `MultiHeadAttention` keeps working and returning the attention weights. Its constructor arguments carry over, except that `dropout_prob` is now `dropout` and the attention `bias` is a keyword of the forward pass; an existing layer is converted by `Attention(mha)`, or its weights loaded with `Flux.loadmodel!`, and `y, α = mha(q, k, v)` becomes `y, α = Flux.attention_with_weights(attn, q, k, v)`. Requires NNlib ≥ 0.9.48 ([#2739](https://github.com/FluxML/Flux.jl/pull/2739)).
- Flux now requires NNlib ≥ 0.9.50 and Optimisers ≥ 0.4.10. With these, Enzyme gradients of `Embedding` with integer indices work, and on a Reactant device `RAdam`, `Rprop`, `AdaGrad`, `AMSGrad`, `AccumGrad` in an `OptimiserChain`, and `adjust!` on an `OptimiserChain` behave as they do eagerly ([#2751](https://github.com/FluxML/Flux.jl/pull/2751)).
- `outputsize` and `@autosize` work with the `elu`, `celu` and `selu` activations of upcoming NNlib versions, which compute them with `expm1` ([#2740](https://github.com/FluxML/Flux.jl/issues/2740)).
- `Dropout` and `AlphaDropout` in train mode draw a fresh mask at every `trainstep!` and `train!` step on a Reactant device. Before, the RNG seed was reset after the first step, so the same mask was reused for the rest of training. Compiling `Dropout` on Reactant works with NNlib ≥ 0.9.50 ([#2744](https://github.com/FluxML/Flux.jl/issues/2744)).
- Fix `using CUDA` sometimes not loading cuDNN on Julia 1.13, which made `gpu` and `gpu_device` fall back to the CPU with a "No functional GPU backend found" warning. It happened whenever Flux's CUDA extension was precompiled before cuDNN ([#2737](https://github.com/FluxML/Flux.jl/pull/2737)).
- On a Reactant device, `trainstep!`, `trainstep_withgradient!` and `train!` no longer reuse a stale compiled step, which silently gave wrong results, after `freeze!`/`thaw!`, `trainmode!`/`testmode!`, a redefinition of the loss, or a batch passing the same array twice, as in `(x, x)` followed by `(x, y)`. A loss that returns an array instead of a scalar is now an error, as it already is without Reactant ([#2744](https://github.com/FluxML/Flux.jl/issues/2744)).
- The recurrent layers `RNN`, `LSTM`, `GRU` and `GRUv3`, their cells and `Recurrence` are now type stable. This makes their gradients up to 30% faster on the CPU and 20% faster on the GPU. It also fixes Mooncake gradients of `LSTM` and `GRU`, which errored (and crashed Julia 1.10), and Enzyme's split reverse mode (`ReverseSplitWithPrimal`), which failed on them with `NonInferredActiveReturn` ([Enzyme.jl#2897](https://github.com/EnzymeAD/Enzyme.jl/issues/2897)) ([#2743](https://github.com/FluxML/Flux.jl/pull/2743)).
- On a Reactant device, `trainstep!`, `trainstep_withgradient!` and `train!` now throw an `ArgumentError` up front when given an AD backend other than `AutoEnzyme()`, the only one supported there. Previously Flux tried to compile the step through that backend and failed with obscure errors, such as "Scalar indexing is disallowed" for `AutoZygote()` or a stack overflow for `AutoMooncake()` ([#2744](https://github.com/FluxML/Flux.jl/issues/2744)).
- On a Reactant device, the compiled training step now skips the update when the loss is not finite, as eager `trainstep!` does. When `train!` stops with a `DomainError`, the parameters and optimiser state are left as they were before the bad batch, instead of already holding `NaN`s ([#2744](https://github.com/FluxML/Flux.jl/issues/2744)).
- `Flux.loadmodel!` loads a saved optimiser state into one from `Flux.setup`, to resume training from a checkpoint. It copies the scalar state too, such as `Adam`'s bias-correction terms, which it previously skipped silently, so that training resumed with the wrong step sizes. Each rule is kept, which also loads a state saved on the host into one set up for a model on a Reactant device ([#2744](https://github.com/FluxML/Flux.jl/issues/2744)).
- On a Reactant device, `trainstep!`, `trainstep_withgradient!` and `train!` throw an `ArgumentError` for an optimiser state not set up for the device, such as a host state moved with `|> dev`. With `Adam` this failed to compile, with `RAdam` it trained with the wrong step sizes, and with `Descent` it ignored later calls to `adjust!` ([#2744](https://github.com/FluxML/Flux.jl/issues/2744)).

## v0.16.12 (14 September 2026)

- `@layer :named MyModel` is a new show option that displays fieldnames in the expanded pretty-print (e.g. `cell = RNNCell(...)` instead of just `RNNCell(...)`) ([#2543](https://github.com/FluxML/Flux.jl/issues/2543)).
- `Dense` now handles inputs with a zero-sized batch dimension (e.g. `Dense(4 => 5)(randn(Float32, 4, 0, 6))`), returning a correctly-shaped empty array instead of an error or a wrong shape ([#2407](https://github.com/FluxML/Flux.jl/issues/2407)).
- Fix stack overflow when applying f16/f32/f64 or cpu/gpu to empty structs declared with Flux.@layer ([#2670](https://github.com/FluxML/Flux.jl/pull/2670)).
- `Conv`, `CrossCor` and `DepthwiseConv` now accept a `pad_mode` keyword that controls how the border is filled: `:zeros` (default), `:circular`, `:reflect`, `:replicate`, or `:symmetric`. This mirrors PyTorch's `padding_mode` and adds support for periodic (circular) and other non-zero paddings ([#1917](https://github.com/FluxML/Flux.jl/issues/1917)).
- The convolution and pooling layers now accept `pad=:same` as a shorthand for same-size padding. `SamePad()` still works but is soft-deprecated in favour of the `:same` symbol ([#1917](https://github.com/FluxML/Flux.jl/issues/1917)).
- **Fix, changed behavior:** for even-sized filters, `:same`/`SamePad()` now places the extra (odd) padding element on the right/bottom instead of the left/top, matching PyTorch and TensorFlow. This makes strided `Conv` and `ConvTranspose` proper adjoints of each other and produces reversal-symmetric outputs. Outputs of even-filter same-padded convolutions shift by one pixel compared to earlier versions ([#2431](https://github.com/FluxML/Flux.jl/issues/2431)).
- `BatchNorm` in training mode now throws a clear error when a channel sees only a single value (e.g. batch size 1 with no spatial dimensions), instead of silently corrupting the running variance with `NaN`. The batch variance of a single value is undefined; this matches PyTorch's behavior (`testmode!` inference with batch size 1 is unaffected) ([#1992](https://github.com/FluxML/Flux.jl/issues/1992)).

## v0.16.11 (6 August 2026)

- `Embedding` now accepts a `padding_idx` keyword. The embedding vector at that index is initialized to zero and its gradient is masked, so it stays fixed during training — useful for a padding token in variable-length sequences ([#2684](https://github.com/FluxML/Flux.jl/issues/2684)).
- `Flux.withgradient(f, model::Duplicated, ...)` (the Enzyme backend) now supports **auxiliary outputs**, matching the Zygote and Mooncake backends: if `f` returns a `Tuple` or `NamedTuple` whose first element is the scalar loss, the gradient is taken of the loss alone and the whole output is returned as `val` ([#2710](https://github.com/FluxML/Flux.jl/pull/2710)).
- Added the public single-step training primitives `Flux.trainstep!(loss, [adtype,] model, batch::Tuple, opt_state) -> loss` and `Flux.trainstep_withgradient!(...) -> (loss, grad)`, which run one optimisation step (gradient + in-place optimiser update). `train!` is now a loop built on top of `trainstep!`, and is compiled and cached automatically on a Reactant device ([#2709](https://github.com/FluxML/Flux.jl/pull/2709)).
- `Flux.trainstep!`, `Flux.trainstep_withgradient!` and `train!` support **auxiliary loss outputs**: if the loss returns a `Tuple` or `NamedTuple` whose first element is the scalar loss, the gradient is taken of the loss alone and the whole value is returned (handy for logging metrics computed in the forward pass). Works on every backend, including Reactant ([#2711](https://github.com/FluxML/Flux.jl/pull/2711)).
- `train!` now automatically compiles the whole training step (forward pass, Enzyme reverse pass and optimiser update) into a single XLA executable when the model lives on a [Reactant](https://github.com/EnzymeAD/Reactant.jl) device. The compiled step is cached and reused across `train!` calls and batch shapes, so multi-epoch training does not recompile. Move the model to the device and call `Flux.setup` before `train!`, and pass device-resident data (e.g. `data .|> reactant_device()`). Requires `Optimisers ≥ 0.4.8` ([#2707](https://github.com/FluxML/Flux.jl/pull/2707)).
- `f16` and `bf16` now convert `BatchNorm`, `InstanceNorm` and `GroupNorm` in **mixed precision**: their statistics and affine parameters are kept in `Float32` while the data flows in half precision. This matches what the NNlib (cuDNN) normalization kernels require for half-precision feature maps, so half-precision `BatchNorm` now works on the GPU. `LayerNorm` is still converted fully (it wraps `NNlib.normalise`, which has no such requirement). Note the behavior change: these layers' parameters are no longer downcast to `Float16`/`BFloat16`. Also adds bf16 `Conv`/pooling GPU coverage. Requires NNlib ≥ 0.9.42 ([#2700](https://github.com/FluxML/Flux.jl/pull/2700)).
- The normalization layers `BatchNorm`, `InstanceNorm`, `GroupNorm` and `LayerNorm` now delegate their forward pass to the functional operators `NNlib.batchnorm`, `NNlib.instancenorm`, `NNlib.groupnorm` and `NNlib.normalise` (which requires NNlib v0.9.41): the normalization logic now lives in NNlib and is shared across the ecosystem. As a side effect, `LayerNorm` and `Flux.normalise` now add `eps` (rather than `eps^2`) to the variance for numerical stability, matching the other normalization layers ([#2701](https://github.com/FluxML/Flux.jl/pull/2701)).
- `Flux.normalise` is now a thin wrapper around `NNlib.normalise`, which should be preferred ([#2701](https://github.com/FluxML/Flux.jl/pull/2701)).
- `train!` now paces garbage collection adaptively instead of caching allocations across steps, keeping peak GPU memory at the working set for allocation-heavy models such as conv nets ([#2695](https://github.com/FluxML/Flux.jl/pull/2695), [#2697](https://github.com/FluxML/Flux.jl/pull/2697)).

## v0.16.10 (17 April 2026)

- `train!` now uses the GPUArrays caching allocator to reuse memory across steps ([#2665](https://github.com/FluxML/Flux.jl/pull/2665)).
- All documented methods are now marked `public` ([#2662](https://github.com/FluxML/Flux.jl/pull/2662)).
- Fixed `Flux.gradient` with the `AutoZygote()` backend ([#2664](https://github.com/FluxML/Flux.jl/pull/2664)).
- Switched to `ParallelTestRunner.jl` for parallel test execution ([#2656](https://github.com/FluxML/Flux.jl/pull/2656)).

## v0.16.9 (1 February 2026)

- Support for `Mooncake` 0.5 ([#2653](https://github.com/FluxML/Flux.jl/pull/2653)).

## v0.16.8 (23 January 2026)

- `Flux.gradient` and `Flux.withgradient` now accept an AD backend argument such as `AutoEnzyme()` and `AutoMooncake()` ([#2645](https://github.com/FluxML/Flux.jl/pull/2645)), and `train!` likewise accepts an AD type ([#2650](https://github.com/FluxML/Flux.jl/pull/2650)).
- The default `init_score` for `early_stopping` is now `Inf` instead of `0`, to prevent unexpected behavior when the defaults are not modified ([#2642](https://github.com/FluxML/Flux.jl/pull/2642)).

## v0.16.7 (9 December 2025)

- Minor documentation fixes and compatibility updates for downstream packages.

## v0.16.6 (8 December 2025)

- Minor dependency bumps and CI updates.

## v0.16.5 (23 July 2025)

- **Fix** for `unsafe_free!` failure with certain `CuArray` configurations.
- Bumped `AMDGPU` compatibility in the weak dependencies.
- Fixed typos in the legacy tutorials documentation.

## v0.16.4 (2 June 2025)

- Added shape validation for convolution weight tensors.
- Fixed missing imports in `FluxMPIExt`.
- Fixed recurrent docstrings and pooling layer loading.
- Fixed type piracy that broke moving a `DataLoader` to a device, e.g. `gpu_device()(dataloader)` ([#2608](https://github.com/FluxML/Flux.jl/pull/2608)).

## v0.16.3 (6 February 2025)

- **Fix** for `cpu(dataloader)` behavior.
- Fixed a "infinite time of gradient" edge case and several data-loading issues.

## v0.16.2 (21 January 2025)

- **Fixes** around new gradients, precompilation on Julia 1.12, and export issues.

## v0.16.1 (13 January 2025)

- Added a "return state" option to recurrent layers.
- Test fixes for Enzyme and Reactant forward/reverse passes, plus documentation updates.

## v0.16.0 (15 December 2024)

This release has a single **breaking change**:

- The recurrent cells `RNNCell`, `LSTMCell`, and `GRUCell` forward has been changed to
  $y_t, state_t = cell(x_t, state_{t-1})$. Previously, it was $state_t = cell(x_t, state_{t-1})$.

Other highlights include:
- Added `WeightNorm` normalization layer.
- Added `Recurrence` layer, turning a recurrent layer into a layer processing the entire sequence at once.

## v0.15.2 (11 December 2024)

- Hotfix for the `LSTM` output ([#2547](https://github.com/FluxML/Flux.jl/pull/2547)).

## v0.15.1 (10 December 2024)

- Added an `initialstates` function to recurrent layers ([#2541](https://github.com/FluxML/Flux.jl/pull/2541)).
- Fixed `Flux.@functor` ([#2546](https://github.com/FluxML/Flux.jl/pull/2546)).

## v0.15.0 (5 December 2024)

This release includes two **breaking changes**:

- The recurrent layers have been thoroughly revised. See below and read the [documentation](https://fluxml.ai/Flux.jl/v0.15/guide/models/recurrence/) for details.
- Flux now defines and exports its own gradient function. Consequently, using gradient in an unqualified manner (e.g., after `using Flux, Zygote`) could result in an ambiguity error.

The most significant updates and deprecations are as follows:
- Recurrent layers have undergone a complete redesign in [#2500](https://github.com/FluxML/Flux.jl/pull/2500).
  - `RNNCell`, `LSTMCell`, and `GRUCell` are now exported and provide functionality for single time-step processing: `rnncell(x_t, h_t) -> h_{t+1}`.
  - `RNN`, `LSTM`, and `GRU` no longer store the hidden state internally, it has to be explicitely passed to the layer. Moreover, they now process entire sequences at once, rather than one element at a time: `rnn(x, h) -> h′`.
  - The `Recur` wrapper has been deprecated and removed.
  - The `reset!` function has also been removed; state management is now entirely up to the user.
- The `Flux.Optimise` module has been deprecated in favor of the Optimisers.jl package.
  Now Flux re-exports the optimisers from Optimisers.jl. Most users will be uneffected by this change.
  The module is still available for now, but will be removed in a future release.
- Most Flux layers will [re-use memory via `NNlib.bias_act!`](https://github.com/FluxML/Flux.jl/pull/2327), when possible.
- `Parallel` now broadcasts over multiple inputs, so `Parallel(+, f)(x, y, z)` applies `f` to each argument, and `Chain(identity, Parallel(+, f))(x, y, z)` works as well ([#2393](https://github.com/FluxML/Flux.jl/pull/2393)).
- Further support for Enzyme.jl, via methods of `Flux.gradient(loss, Duplicated(model))`.
  Flux now owns & exports `gradient` and `withgradient`, but without `Duplicated` this still defaults to calling Zygote.jl.
- `Flux.params` has been deprecated. Use Zygote's explicit differentiation instead,
  `gradient(m -> loss(m, x, y), model)`, or use `Flux.trainables(model)` to get the trainable parameters.
- Flux now requires Functors.jl v0.5. This new release of Functors assumes all types to be functors by default. Therefore, applying `Flux.@layer` or `Functors.@functor` to a type is no longer strictly necessary for Flux's models. However, it is still recommended to use `@layer Model` for additional functionality like pretty printing.
- `@layer Model` now behaves the same as `@layer :expand Model`, which means that the model is expanded into its sublayers (if there are any) when printed. To force compact printing, use `@layer :noexpand Model`.

## v0.14.25 (3 November 2024)

- Reintroduced `FluxCUDAAdaptor` and related adaptors to smooth out the transition to MLDataDevices.jl ([#2512](https://github.com/FluxML/Flux.jl/pull/2512)).

## v0.14.24 (1 November 2024)

- Properly deprecated the old `GPU_BACKEND` preference mechanism ([#2511](https://github.com/FluxML/Flux.jl/pull/2511)).

## v0.14.23 (29 October 2024)

- Added `lecun_normal` weight initialization ([#2311](https://github.com/FluxML/Flux.jl/pull/2311)).
- `gpu(x)` now dispatches to `gpu_device()` from MLDataDevices.jl ([#2502](https://github.com/FluxML/Flux.jl/pull/2502)).

## v0.14.22 (12 October 2024)

- Data movement between devices is now provided by [MLDataDevices.jl](https://github.com/LuxDL/MLDataDevices.jl) ([#2492](https://github.com/FluxML/Flux.jl/pull/2492)).

## v0.14.21 (4 October 2024)

- CUDA is now an optional dependency for `FluxMPIExt` ([#2488](https://github.com/FluxML/Flux.jl/pull/2488)).

## v0.14.20 (20 September 2024)

- Added [support for distributed data parallel training](https://github.com/FluxML/Flux.jl/pull/2464). MPI and NCCL backends are available with the `FluxMPIExt` and `FluxMPINCCLExt` extensions respectively.

## v0.14.19 (14 August 2024)

- Allow loading a `ConvTranspose` state saved without the `.outpad` field ([#2477](https://github.com/FluxML/Flux.jl/pull/2477)).

## v0.14.18 (9 August 2024)

- Dependency updates.

## v0.14.17 (3 August 2024)

- Added [support for Enzyme](https://github.com/FluxML/Flux.jl/pull/2446) in `Flux.train!`, with a flag to select Enzyme as the training AD backend.

## v0.14.16 (17 June 2024)

- Allow `BatchNorm` on CUDA with `track_stats=false` ([#2427](https://github.com/FluxML/Flux.jl/pull/2427)).
- Removed the BSON dependency ([#2458](https://github.com/FluxML/Flux.jl/pull/2458)).

## v0.14.15 (21 March 2024)

- Restored some support for `Tracker.jl` ([#2387](https://github.com/FluxML/Flux.jl/pull/2387)).
- Improved error messages for convolution layers ([#2404](https://github.com/FluxML/Flux.jl/pull/2404)).

## v0.14.14 (18 March 2024)

- Use `LazyString` in deprecation warnings, plus precompilation and performance fixes ([#2400](https://github.com/FluxML/Flux.jl/pull/2400)).

## v0.14.13 (7 March 2024)

- New macro `Flux.@layer` which should be used in place of `@functor`.
  This also adds `show` methods for pretty printing.

## v0.14.12 (21 February 2024)

- New `SignDecay` optimiser, like `WeightDecay` but for L1 norm ([#2377](https://github.com/FluxML/Flux.jl/pull/2377)).

## v0.14.11 (31 January 2024)

- Support for `Adapt.jl` v4 ([#2374](https://github.com/FluxML/Flux.jl/pull/2374)).

## v0.14.10 (24 January 2024)

- Restored the missing `rng_from_array(::CuArray)` method ([#2369](https://github.com/FluxML/Flux.jl/pull/2369)).

## v0.14.9 (8 January 2024)

- Restored type stability of `conv_transpose_dims` ([#2365](https://github.com/FluxML/Flux.jl/pull/2365)).
- Hotfix for the new `OneElement` on GPU ([#2368](https://github.com/FluxML/Flux.jl/pull/2368)).

## v0.14.8 (28 December 2023)

- Non-differentiable shape handling in normalization layers ([#2363](https://github.com/FluxML/Flux.jl/pull/2363)).
- Use the stable API for AMDGPU RNG conversion ([#2360](https://github.com/FluxML/Flux.jl/pull/2360)).

## v0.14.7 (5 December 2023)

- Use the new `public` keyword to mark the public API ([#2342](https://github.com/FluxML/Flux.jl/pull/2342)).

## v0.14.6 (13 September 2023)

- Updated CUDA and AMDGPU compatibility bounds ([#2338](https://github.com/FluxML/Flux.jl/pull/2338)).
- Moved the codebase to 4-space indentation.

## v0.14.5 (7 September 2023)

- Renamed the `"AMD"` GPU backend to `"AMDGPU"` ([#2328](https://github.com/FluxML/Flux.jl/pull/2328)).

## v0.14.4 (28 August 2023)

- Added `get_device("Metal")` support with more informative error messages ([#2319](https://github.com/FluxML/Flux.jl/pull/2319)).

## v0.14.3 (27 August 2023)

- Implemented an interface for data transfer across GPU devices ([#2308](https://github.com/FluxML/Flux.jl/pull/2308)).
- Allow `Optimisers.jl` v0.3 ([#2318](https://github.com/FluxML/Flux.jl/pull/2318)).

## v0.14.2 (7 August 2023)

- Added device objects for selecting GPU backends, defaulting to CPU when none is available ([#2297](https://github.com/FluxML/Flux.jl/pull/2297)).

## v0.14.1 (19 July 2023)

- `gpu(x)` now returns `x` unchanged when no GPU backend is loaded ([#2295](https://github.com/FluxML/Flux.jl/pull/2295)).

## v0.14.0 (14 July 2023)

- Flux now requires julia v1.9 or later.
- CUDA.jl is not a hard dependency anymore. Support is now provided through the extension mechanism, by loading `using Flux, CUDA`.
  The package cuDNN.jl also needs to be installed in the environment. (You will get instructions if this is missing.)
- After a deprecations cycle, the macro `@epochs` and the functions `Flux.stop`, `Flux.skip`, `Flux.zeros`, `Flux.ones` have been removed.

## v0.13.17 (17 June 2023)

- Apple's Metal GPU acceleration preliminary support via the extension mechanism.

## v0.13.16 (5 May 2023)

- Most greek-letter keyword arguments are deprecated in favour of ascii.
  Thus `LayerNorm(3; ϵ=1e-4)` (not `ε`!) should become `LayerNorm(3; eps=1e-4)`.
- `DataLoader(...) |> gpu` will now produce a special iterator, moving each batch as needed,
  instead of giving an error.
- Added `Flux.state` returning the internal state of the model for serialization.

## v0.13.15 (18 April 2023)

- Added [MultiHeadAttention](https://github.com/FluxML/Flux.jl/pull/2146) layer.
- `f16, f32, f64` now specifically target floating point arrays (i.e. integers arrays and other types are preserved).
- `f16, f32, f64` can now handle `Complex{<:AbstractFloat}` arrays.
- Added `EmbeddingBag` layer.

## v0.13.14 (10 March 2023)

- Fixed various deprecation warnings, from `Zygone.@nograd` and `Vararg`.
- Initial support for `AMDGPU` via extension mechanism.
- Add `gpu_backend` preference to select GPU backend using `LocalPreference.toml`.
- Add `Flux.gpu_backend!` method to switch between GPU backends.

## v0.13.13 (18 February 2023)

- Added `f16` which changes precision to `Float16`, recursively.
- Most layers standardise their input to `eltype(layer.weight)`, [#2156](https://github.com/FluxML/Flux.jl/pull/2156),
  to limit the cost of accidental Float64 promotion.
- Friendlier errors from size mismatches [#2176](https://github.com/FluxML/Flux.jl/pull/2176).

## v0.13.12 (4 February 2023)

- CUDA.jl 4.0 compatibility.
- Use `dropout` from NNlib as back-end for `Dropout` layer.

## v0.13.9 (30 November 2022)

- New method of `train!` using Zygote's "explicit" mode. Part of a move away from "implicit" `Params`.
- Added [Flux.setup](https://github.com/FluxML/Flux.jl/pull/2082), which is `Optimisers.setup` with extra checks,
  and translation from deprecated "implicit" optimisers like `Flux.Optimise.Adam` to new ones from Optimisers.jl.

## v0.13.7 (29 October 2022)

- Added [`@autosize` macro](https://github.com/FluxML/Flux.jl/pull/2078), as another way to use `outputsize`.
- Export `Embedding`.

## v0.13.6 (13 September 2022)

- Use the package [OneHotArrays.jl](https://github.com/FluxML/OneHotArrays.jl) instead of having the same code here.
- Added [`@autosize` macro](https://github.com/FluxML/Flux.jl/pull/2078)

## v0.13.4 (5 July 2022)

- Added [`PairwiseFusion` layer](https://github.com/FluxML/Flux.jl/pull/1983)
- Re-name `ADAM` to `Adam`, etc (with deprecations).

## v0.13.0 (8 April 2022)

- After a deprecations cycle, the datasets in `Flux.Data` have
  been removed in favour of [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl).
- `params` is not exported anymore since it is a common name and is also exported by Distributions.jl
- `flatten` is not exported anymore due to clash with `Iterators.flatten`.
- Remove Juno.jl progress bar support as it is now obsolete.
- `Dropout` gained improved compatibility with Int and Complex arrays and is now twice-differentiable.
- Notation `Dense(2 => 3, σ)` for channels matches `Conv`; the equivalent `Dense(2, 3, σ)` still works.
- Many utily functions and the `DataLoader` are [now provided by MLUtils.jl](https://github.com/FluxML/Flux.jl/pull/1874).
- The DataLoader is now compatible with generic dataset types implementing `MLUtils.numobs` and `MLUtils.getobs`.
- Added [truncated normal initialisation](https://github.com/FluxML/Flux.jl/pull/1877) of weights.
- The `Flux.Diagonal` layer is now called `Scale`, and accepts an activation function.
- `loadparams!` is replaced by [`loadmodel!`](https://github.com/FluxML/Flux.jl/pull/1875) which copies trainable + non-trainable parameters and performs more thorough structural checking

## v0.12.10 (7 April 2022)

- `Dropout`/`AlphaDropout` now supports [user-specified RNGs](https://github.com/FluxML/Flux.jl/pull/1838)

## v0.12.9 (27 January 2022)

- Fixed incorrect output and added GPU compatibility for [AlphaDropout](https://github.com/FluxML/Flux.jl/pull/1781).
- Add trilinear [Upsample layer](https://github.com/FluxML/Flux.jl/pull/1792).
- Improved [performance of RNNs](https://github.com/FluxML/Flux.jl/pull/1761)
- Optimisers now accept an `ϵ` argument
- [Improved handling of complex values inputs](https://github.com/FluxML/Flux.jl/pull/1776) while training
- Fixed [AlphaDropout](https://github.com/FluxML/Flux.jl/pull/1781)

## v0.12.8 (28 October 2021)

- Optimised inference and gradient calculation of OneHotMatrix[pr](https://github.com/FluxML/Flux.jl/pull/1756)

## v0.12.7 (29 September 2021)

- Added support for [`GRUv3`](https://github.com/FluxML/Flux.jl/pull/1675)
- The layers within `Chain` and `Parallel` may now [have names](https://github.com/FluxML/Flux.jl/issues/1680).

## v0.12.5 (13 July 2021)

- Added option to configure [`groups`](https://github.com/FluxML/Flux.jl/pull/1531) in `Conv`.
- REPL printing via [`show`](https://github.com/FluxML/Flux.jl/pull/1467) displays parameter counts.

## v0.12.4 (1 June 2021)

- Implemented an [`Embedding layer`](https://github.com/FluxML/Flux.jl/pull/1516)
  based on `NNlib.gather` and `NNlib.scatter`.

## v0.12.1 - v0.12.3 (April 2021)

- CUDA.jl 3.0 support
- Bug fixes and optimizations.

## v0.12.0 (27 March 2021)

- Add [identity_init](https://github.com/FluxML/Flux.jl/pull/1524).
- Add [Orthogonal Matrix initialization](https://github.com/FluxML/Flux.jl/pull/1496) as described in [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120).
- Added [Focal Loss function](https://github.com/FluxML/Flux.jl/pull/1489) to Losses module
- The Dense layer now supports inputs with [multiple batch dimensions](https://github.com/FluxML/Flux.jl/pull/1405).
- Dense and Conv layers no longer perform  [implicit type conversion](https://github.com/FluxML/Flux.jl/pull/1394).
- The keyword `initW` is of Dense layers is now `init`, to agree with convolutional layers.
- Excise datasets in favour of other providers in the julia ecosystem.
- Added option to set `bias` to [false](https://github.com/FluxML/Flux.jl/pull/1379) to eliminating `bias` from being trained.
- Add [CTC loss function](https://github.com/FluxML/Flux.jl/pull/1287) to Losses module
- Removed kwarg only constructors for [`convolutional layers`](https://github.com/FluxML/Flux.jl/pull/1379).
- Add [sparse initialization](https://github.com/FluxML/Flux.jl/pull/1454) as described in [Deep learning via Hessian-free optimization](https://dl.acm.org/doi/abs/10.5555/3104322.3104416).
- Moved GPU CI to use buildkite instead of GitLab
- New [`Parallel` layer](https://github.com/FluxML/Flux.jl/pull/1462) adds inception module-like building blocks.
- Feature additions and bug fixes for BatchNorm, LayerNorm, InstanceNorm, and GroupNorm [normalization layers](https://github.com/FluxML/Flux.jl/pull/1397)
- Added [Upsample and PixelShuffle layers](https://github.com/FluxML/Flux.jl/pull/1468)
- End of deprecation cycle: loss functions cannot be accessed directly from `Flux` anymore, they live in the `Flux.Losses` module.
  All loss functions perform `mean` aggregation by default.

## v0.11.2 (6 November 2020)

- Adds the [AdaBelief](https://arxiv.org/abs/2010.07468) optimiser.
- Other new features and bug fixes (see GitHub releases page)

## v0.11.0 (10 July 2020)

- Moved CUDA compatibility to use [CUDA.jl instead of CuArrays.jl](https://github.com/FluxML/Flux.jl/pull/1204)
- Add [kaiming initialization](https://arxiv.org/abs/1502.01852) methods: [kaiming_uniform and kaiming_normal](https://github.com/FluxML/Flux.jl/pull/1243)
- Use `DataLoader` with `NamedTuple`s, so that tensors can be accessed [by name](https://github.com/FluxML/Flux.jl/pull/1221).
- Error if Dense layers weights and biases are [not arrays](https://github.com/FluxML/Flux.jl/pull/1218).
- Add [Adaptive Pooling](https://github.com/FluxML/Flux.jl/pull/1239) in Flux layers.
- Change to `DataLoader`'s [constructor](https://github.com/FluxML/Flux.jl/pull/1152)
- Uniform loss [interface](https://github.com/FluxML/Flux.jl/pull/1150)
- Loss functions now live in the `Flux.Losses` [module](https://github.com/FluxML/Flux.jl/pull/1264)
- Optimistic ADAM (OADAM) optimiser for [adversarial training](https://github.com/FluxML/Flux.jl/pull/1246).
- Add option for [same padding](https://github.com/FluxML/Flux.jl/pull/901) to conv and pooling layers by setting `pad=SamePad()`.
- Added option to set `bias` to [Flux.Zeros](https://github.com/FluxML/Flux.jl/pull/873) to eliminating `bias` from being trained.
- Added `GlobalMaxPool` and `GlobalMeanPool` [layers](https://github.com/FluxML/Flux.jl/pull/950) for performing global pooling operations.
- Added `ClipValue` and `ClipNorm` in this [pr](https://github.com/FluxML/Flux.jl/pull/1133) to `Flux.Optimise` to provide a cleaner API for gradient clipping.
- Added new kwarg-only [constructors](https://github.com/FluxML/Flux.jl/pull/873) for the various convolutional layers.
- Documented the convolutional layer constructors accepting `weight` and `bias` keyword arguments to supply custom arrays for those fields.
- Testing suite improvements now test for gradients of all layers along with GPU support.
- Functors have now moved to [Functors.jl](https://github.com/FluxML/Flux.jl/pull/1174) to allow for their use outside of Flux.
- Added [helper functions](https://github.com/FluxML/Flux.jl/pull/873) `Flux.convfilter` and `Flux.depthwiseconvfilter` to construct weight arrays for convolutions outside of layer constructors so as to not have to depend on the default layers for custom implementations.
- `dropout` function now has a mandatory [active](https://github.com/FluxML/Flux.jl/pull/1263)
  keyword argument. The `Dropout` struct (whose behavior is left unchanged) is the recommended choice for common usage.
- and many more fixes and additions...

## v0.10.1 - v0.10.4 (2020)

See GitHub's releases.

## v0.10.0 (29 November 2019)

- The default AD engine has switched from [Tracker to Zygote.jl](https://github.com/FluxML/Flux.jl/pull/669)
  - The dependency on Tracker.jl has been removed.
  - This means Flux now does not depend on using a specialised `TrackedArray` type, and can be used with normal Array implementations directly.
  - Tracker compatibility is maintained in most common cases, but Zygote will be the preferred AD backend for Flux from now on.
- The CUDNN wrappers have been [moved from Flux into CuArrays](https://github.com/FluxML/Flux.jl/pull/874), to allow for better supporting the CUDA backend, and improve user experience, not to mention making Flux lean.
- `*crossentropy` functions now [work as expected with CuArrays](https://github.com/FluxML/Flux.jl/pull/926). [PR for binarycrossentropy](https://github.com/FluxML/Flux.jl/pull/940).
- Added [clearer docs](https://github.com/FluxML/Flux.jl/pull/904) around training and the Optimiser interface.
- [Layer initialisations](https://github.com/FluxML/Flux.jl/pull/937) have been improved with a clearer API on how to extend it for other purposes.
- [Better messaging around CUDA availability](https://github.com/FluxML/Flux.jl/pull/924), with hooks to initialize the GPU as default where possible.
- `@treelike` has been formalised as a [functor](https://github.com/FluxML/Flux.jl/pull/865), with an effective deprecation.
- `testmode!` is deprecated in favour of [istraining](https://github.com/FluxML/Flux.jl/pull/669)

## v0.9.0 (29 August 2019)

- [Depthwise convolutional layer API changes](https://github.com/FluxML/Flux.jl/pull/756) from `in => mult` channel specification to `in => out` channel specification, and deprecates implicit `out` constructor.
- New [SkipConnection](https://github.com/FluxML/Flux.jl/pull/446), which can be used to train residual neural network architectures.
- New [RADAM](https://github.com/FluxML/Flux.jl/pull/842) optimiser.

## v0.8.0 (22 March 2019)

- [Dropout now has a `dims` argument for specifying the unbroadcast dimensions.](https://github.com/FluxML/Flux.jl/pull/563)
- New [ConvTranspose layer](https://github.com/FluxML/Flux.jl/pull/311).
- New [Maxout layer](https://github.com/FluxML/Flux.jl/pull/647)
- Datasets are now [hash verified on download](https://github.com/FluxML/Flux.jl/pull/585) to avoid corruption.
- We now [zero the initial state for RNNs](https://github.com/FluxML/Flux.jl/pull/590/).
- [Normalisation can now work on arbitrary `dims`.](https://github.com/FluxML/Flux.jl/pull/592)
- Many docs and bugfixes thanks to @KristofferC and others.
- [NamedTuples now work like Tuples](https://github.com/FluxML/Flux.jl/pull/603) when doing `mapleaves`.
- New "performance tips" [section of the docs](https://github.com/FluxML/Flux.jl/pull/615).
- The training loop is [now more readable](https://github.com/FluxML/Flux.jl/pull/651) and better shows how to use the lower-level APIs.
- New [AlphaDropout](https://github.com/FluxML/Flux.jl/pull/656).
- [Data.Iris](https://github.com/FluxML/Flux.jl/pull/652) makes Fisher's Iris dataset available with `Iris.labels` and `Iris.features`.
- New [InstanceNorm](https://github.com/FluxML/Flux.jl/pull/634), as popularized by [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).
- New [GroupNorm](https://github.com/FluxML/Flux.jl/pull/696), as described in [Group Normalization](https://arxiv.org/abs/1803.08494).
- New [CrossCor](https://github.com/FluxML/Flux.jl/pull/762).

AD Changes:

- `det`, `logdet` and `logabsdet` [now have adjoints](https://github.com/FluxML/Flux.jl/pull/596/files).
- Support for [PermuteDimsArray](https://github.com/FluxML/Flux.jl/pull/576).
- Flux.Tracker is now its [own package](https://github.com/FluxML/Tracker.jl), in preparation for replacing it with Zygote.

## v0.7.0 (16 January 2019)

Despite the heroic efforts of scholars and archeologists, pre-0.7 history is lost to the sands of time.
