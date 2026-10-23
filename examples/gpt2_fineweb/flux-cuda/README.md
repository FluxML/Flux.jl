# GPT-2 pretraining with Flux and CUDA.jl

The [GPT-2 example](../README.md) on [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), without
Reactant: each training step runs eagerly on `CuArray`s, differentiated by Zygote.
[`train.jl`](train.jl) includes the model, data pipeline, HellaSwag evaluation and training loop
of the reference version in [`../flux-reactant`](../flux-reactant/), and only chooses the device,
the precision of matrix multiplications and the memory management.

| file | what |
| --- | --- |
| [`train.jl`](train.jl) | the training script: runs the loop of [`../flux-reactant/training.jl`](../flux-reactant/training.jl) on CUDA |
| [`hellaswag.jl`](hellaswag.jl) | HellaSwag accuracy of a saved checkpoint |
| [`Project.toml`](Project.toml) | the environment: CUDA.jl and cuDNN.jl instead of Reactant |

## Running

```console
$ cd examples/gpt2_fineweb/flux-cuda
$ julia --project=. -e 'using Pkg; Pkg.instantiate()'

$ julia --project=. train.jl --preset tinystories
$ julia --project=. train.jl --preset fineweb
$ julia --project=. train.jl --preset fineweb --resume log/checkpoint_05000.jld2
$ julia --project=. hellaswag.jl log/checkpoint_19073.jld2
```

The options are those of the [Reactant version](../flux-reactant/README.md#running), except that
`--batchsize` defaults to 8 for `fineweb` (see the notes). The tokenized datasets in `../data/` are shared with the other
versions, and the checkpoints are interchangeable with the Reactant version's.

## Results

Measured on an **RTX 5090 (32 GB)**, Julia 1.13, Flux 0.16, CUDA.jl 6.4, cuDNN 9.27. Full training
runs are still to be done.

**`tinystories`** — 0.53 s per step (~125k tokens/s), against 0.17 s with Reactant, and ~90 s for
the first step, against ~160 s. Over the first steps, the training and validation losses agree with
Reactant's to the fourth digit.

**`fineweb`** — ~31k tokens/s (~27 TFLOPS) at micro-batch 8, i.e. ~17 s per step of 2^19 tokens,
measured on micro-batches of synthetic data; see the [comparison](../README.md#performance). The
garbage collection after each micro-batch makes it 1.4× faster (262 ms per micro-batch instead of
365 ms, on average).

## Notes

- **Precision.** `--precision bf16` (the default) is as in the Reactant version: a bfloat16 model,
  `Float32` master weights in the optimizer, and the loss in `Float32`. `tf32` runs `Float32`
  matrix multiplications on TF32 tensor cores (`CUDA.math_mode!(CUDA.FAST_MATH)`), and `f32` keeps
  them exact.
- **Attention.** With cuDNN.jl loaded, the causal `Attention` layers of the bfloat16 model run
  cuDNN's fused flash-attention kernels, forward and backward, so the `(T, T, heads, batch)`
  attention matrix is never materialized. In `Float32` they use NNlib's fused kernel.
- **Memory.** Zygote keeps every intermediate array of the forward pass for the backward pass, and
  CUDA.jl frees GPU memory only when Julia's garbage collector finalizes the arrays. GPT-2 124M
  therefore fits 8 sequences per micro-batch in 32 GB, which is the script's default for `fineweb`
  (the gradient accumulation still makes 2^19 tokens per step). The script also runs an incremental
  garbage collection after each micro-batch (`gc = true` in `train`), which frees the step's arrays
  before the next one allocates.
- **Compilation.** There is no XLA compilation: the first step takes ~1.5 minutes of Julia
  compilation, against ~3 to 12 minutes with Reactant.
- **Automatic differentiation.** Only Zygote, for now. Mooncake 0.5.65 cannot differentiate this
  model on CUDA ([Mooncake.jl#1393](https://github.com/chalk-lab/Mooncake.jl/issues/1393),
  [#1394](https://github.com/chalk-lab/Mooncake.jl/issues/1394),
  [#1395](https://github.com/chalk-lab/Mooncake.jl/issues/1395),
  [NNlib.jl#823](https://github.com/FluxML/NNlib.jl/issues/823)), and the latest Enzyme.jl
  requires LLVM.jl 9, which holds back CUDA.jl.
- **Tied embedding.** The gradient of the output layer reaches the token embedding as a
  `Transpose` of a `CuArray`, which Zygote 0.7.13 adds to the embedding's own gradient through host
  memory ([Zygote.jl#1676](https://github.com/FluxML/Zygote.jl/issues/1676)): a round trip of
  77 MB per micro-batch.
