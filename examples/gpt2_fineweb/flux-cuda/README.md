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

The options are those of the [Reactant version](../flux-reactant/README.md#running). The tokenized
datasets in `../data/` are shared with the other versions, and the checkpoints are interchangeable
with the Reactant version's.

## Results

Measured on an **RTX 5090 (32 GB)**, Julia 1.13, Flux 0.16.14, NNlib 0.9.55, Zygote 0.7.14, CUDA.jl
6.4, cuDNN 9.27, over the first steps of `train.jl`. Full training runs are still to be done.

**`tinystories`** — 0.18 s per step (~360k tokens/s), and ~70 s for the first step. Over the first
steps, the training and validation losses agree with Reactant's to the fourth digit.

**`fineweb`** — ~3.9 s per step of 2^19 tokens (~135k tokens/s, ~115 TFLOPS) at the default
micro-batch of 16, and ~72 s for the first step; see the [comparison](../README.md#performance).
The garbage collection after each micro-batch makes it 1.7× faster (132 ms per micro-batch instead
of 224 ms, on average).

## Notes

- **Precision.** `--precision bf16` (the default) is as in the Reactant version: a bfloat16 model,
  `Float32` master weights in the optimizer, and the loss in `Float32`. `tf32` runs `Float32`
  matrix multiplications on TF32 tensor cores (`CUDA.math_mode!(CUDA.FAST_MATH)`), and `f32` keeps
  them exact.
- **Attention.** With cuDNN.jl loaded, the causal `Attention` layers of the bfloat16 model run
  cuDNN's fused flash-attention kernels, forward and backward, so the `(T, T, heads, batch)`
  attention matrix is never materialized. In `Float32` they use NNlib's fused kernel.
- **Fused kernels.** LayerNorm (`NNlib.layernorm`) and the cross-entropy over the logits run as
  fused kernels, forward and backward, that accumulate in `Float32`. Every other operation is a
  kernel of its own.
- **Memory.** Zygote keeps every intermediate array of the forward pass for the backward pass, and
  CUDA.jl frees GPU memory only when Julia's garbage collector finalizes the arrays. `Flux.trainstep!`
  runs an incremental garbage collection after each micro-batch, which frees the micro-batch's arrays
  before the next one allocates: a micro-batch of 16 sequences of GPT-2 124M then peaks at 22 GB.
  Without it (`gc_interval = 0`), the memory pool fills up to 31 GB and CUDA.jl stalls on full
  collections.
- **Compilation.** There is no XLA compilation: the first step takes ~1.2 minutes of Julia
  compilation, against ~3 to 12 minutes with Reactant.
- **Automatic differentiation.** Only Zygote, for now. Mooncake 0.5.65 cannot differentiate this
  model on CUDA ([Mooncake.jl#1393](https://github.com/chalk-lab/Mooncake.jl/issues/1393),
  [#1394](https://github.com/chalk-lab/Mooncake.jl/issues/1394),
  [#1395](https://github.com/chalk-lab/Mooncake.jl/issues/1395),
  [NNlib.jl#823](https://github.com/FluxML/NNlib.jl/issues/823)), and the latest Enzyme.jl
  requires LLVM.jl 9, which holds back CUDA.jl.
