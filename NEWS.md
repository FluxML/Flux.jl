# v0.9.0
* [Depthwise convolutional layer API changes](https://github.com/FluxML/Flux.jl/pull/756) from `in => mult` channel specification to `in => out` channel specification, and deprecates implicit `out` constructor.
* New [SkipConnection](https://github.com/FluxML/Flux.jl/pull/446), which can be used to train residual neural network architectures.

# v0.8.0

* [Dropout now has a `dims` argument for specifying the unbroadcast dimensions.](https://github.com/FluxML/Flux.jl/pull/563)
* New [ConvTranspose layer](https://github.com/FluxML/Flux.jl/pull/311).
* New [Maxout layer](https://github.com/FluxML/Flux.jl/pull/647)
* Datasets are now [hash verified on download](https://github.com/FluxML/Flux.jl/pull/585) to avoid corruption.
* We now [zero the initial state for RNNs](https://github.com/FluxML/Flux.jl/pull/590/).
* [Normalisation can now work on arbitrary `dims`.](https://github.com/FluxML/Flux.jl/pull/592)
* Many docs and bugfixes thanks to @KristofferC and others.
* [NamedTuples now work like Tuples](https://github.com/FluxML/Flux.jl/pull/603) when doing `mapleaves`.
* New "performance tips" [section of the docs](https://github.com/FluxML/Flux.jl/pull/615).
* The training loop is [now more readable](https://github.com/FluxML/Flux.jl/pull/651) and better shows how to use the lower-level APIs.
* New [AlphaDropout](https://github.com/FluxML/Flux.jl/pull/656).
* [Data.Iris](https://github.com/FluxML/Flux.jl/pull/652) makes Fisher's Iris dataset available with `Iris.labels` and `Iris.features`.
* New [InstanceNorm](https://github.com/FluxML/Flux.jl/pull/634), as popularized by [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).
* New [GroupNorm](https://github.com/FluxML/Flux.jl/pull/696), as described in [Group Normalization](https://arxiv.org/abs/1803.08494).
* New [CrossCor](https://github.com/FluxML/Flux.jl/pull/762).

AD Changes:

* `det`, `logdet` and `logabsdet` [now have adjoints](https://github.com/FluxML/Flux.jl/pull/596/files).
* Support for [PermuteDimsArray](https://github.com/FluxML/Flux.jl/pull/576).
* Flux.Tracker is now its [own package](https://github.com/FluxML/Tracker.jl), in preparation for replacing it with Zygote.

# v0.7.0

Despite the heroic efforts of scholars and archeologists, pre-0.7 history is lost to the sands of time.
