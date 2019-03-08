# v0.8.0

* New [ConvTranspose layer](https://github.com/FluxML/Flux.jl/pull/311).
* Datasets are now [hash verified on download](https://github.com/FluxML/Flux.jl/pull/585) to avoid corruption.
* We now [zero the initial state for RNNs](https://github.com/FluxML/Flux.jl/pull/590/).
* [Normalisation can now work on arbitrary `dims`.](https://github.com/FluxML/Flux.jl/pull/592)
* Many docs and bugfixes thanks to @KristofferC and others.
* [NamedTuples now work like Tuples](https://github.com/FluxML/Flux.jl/pull/603) when doing `mapleaves`.
* New "performance tips" [section of the docs](https://github.com/FluxML/Flux.jl/pull/615).
* The training loop is [now more readable](https://github.com/FluxML/Flux.jl/pull/651) and better shows how to use the lower-level APIs.
* New [AlphaDropout](https://github.com/FluxML/Flux.jl/pull/656).

AD Changes:

* `det`, `logdet` and `logabsdet` [now have adjoints](https://github.com/FluxML/Flux.jl/pull/596/files).
* Support for [PermuteDimsArray](https://github.com/FluxML/Flux.jl/pull/576).
* Flux.Tracker is now its [own package](https://github.com/FluxML/Tracker.jl), in preparation for replacing it with Zygote.

# v0.7.0

Despite the heroic efforts of scholars and archeologists, pre-0.7 history is lost to the sands of time.
