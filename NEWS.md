# Flux Release Notes

See also [github's page](https://github.com/FluxML/Flux.jl/releases) for a complete list of PRs merged before each release.

## Unreleased

- Added the public `Flux.train_step!(loss, [adtype,] model, batch::Tuple, opt_state) -> (loss, grad)`, which runs a single optimisation step (gradient + in-place optimiser update) and returns the loss and the gradient with respect to the model. `train!` is now a loop built on top of it, and `train_step!` uses the cached compiled step automatically when the model lives on a Reactant device (the cached step is dropped once the model is garbage-collected) ([#2709](https://github.com/FluxML/Flux.jl/pull/2709)).
- `train!` now automatically compiles the whole training step (forward pass, Enzyme reverse pass and optimiser update) into a single XLA executable when the model lives on a [Reactant](https://github.com/EnzymeAD/Reactant.jl) device — no code change beyond moving the model to the device. The compiled step is cached and reused across `train!` calls and batch shapes, so multi-epoch training does not recompile. Move the model to the device and call `Flux.setup` before `train!`, and pass device-resident data (e.g. `data .|> reactant_device()`). Requires `Optimisers ≥ 0.4.8` ([#2707](https://github.com/FluxML/Flux.jl/pull/2707)).
- `f16` and `bf16` now convert `BatchNorm`, `InstanceNorm` and `GroupNorm` in **mixed precision**: their statistics and affine parameters are kept in `Float32` while the data flows in half precision. This matches what the NNlib (cuDNN) normalization kernels require for half-precision feature maps, so half-precision `BatchNorm` now works on the GPU. `LayerNorm` is still converted fully (it wraps `NNlib.normalise`, which has no such requirement). Note the behavior change: these layers' parameters are no longer downcast to `Float16`/`BFloat16`. Also adds bf16 `Conv`/pooling GPU coverage. Requires NNlib ≥ 0.9.42 ([#2700](https://github.com/FluxML/Flux.jl/pull/2700)).
- The normalization layers `BatchNorm`, `InstanceNorm`, `GroupNorm` and `LayerNorm` now delegate their forward pass to the functional operators `NNlib.batchnorm`, `NNlib.instancenorm`, `NNlib.groupnorm` and `NNlib.normalise` (which requires NNlib v0.9.41): the normalization logic now lives in NNlib and is shared across the ecosystem. As a side effect, `LayerNorm` and `Flux.normalise` now add `eps` (rather than `eps^2`) to the variance for numerical stability, matching the other normalization layers ([#2701](https://github.com/FluxML/Flux.jl/pull/2701)).
- `Flux.normalise` is now a thin wrapper around `NNlib.normalise`, which should be preferred ([#2701](https://github.com/FluxML/Flux.jl/pull/2701)).
- Removed the `FluxCUDAcuDNNExt` extension and the `cuDNN` dependency: cuDNN-accelerated `BatchNorm` on the GPU is now provided by NNlib and selected automatically for `CuArray`s ([#2701](https://github.com/FluxML/Flux.jl/pull/2701)). On AMDGPU the MIOpen `BatchNorm` fast path was likewise removed; `BatchNorm` uses NNlib's generic path there until a MIOpen fast path lands in NNlib ([NNlib.jl#752](https://github.com/FluxML/NNlib.jl/issues/752)).
- `train!` now defaults to `caching_allocator = false, gc_interval = :auto`: it runs without the cross-step allocation cache and instead paces an incremental garbage collection adaptively, choosing the cadence from step timing. This keeps GPU memory at the working set for allocation-heavy models (the caching allocator pins a step's allocations, which inflates peak memory and could OOM conv nets) while still bounding the ever-growing reserved memory of [#2523](https://github.com/FluxML/Flux.jl/issues/2523). See [#2695](https://github.com/FluxML/Flux.jl/pull/2695) and [#2697](https://github.com/FluxML/Flux.jl/pull/2697).

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
