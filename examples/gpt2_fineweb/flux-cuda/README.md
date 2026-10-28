# GPT-2 pretraining with Flux and CUDA.jl

The [GPT-2 example](../README.md) on [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), without
Reactant: each training step runs eagerly on `CuArray`s, differentiated by Zygote.
[`train.jl`](train.jl) includes the model, data pipeline, HellaSwag evaluation and training loop
of the reference version in [`../flux-reactant`](../flux-reactant/), and only chooses the device
and the precision of matrix multiplications.

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

The [shared comparison](../README.md#performance) records the current versions, timings and
memory measurements for eight steps of GPT-2 124M on the RTX 5090. Full CUDA training runs are
still to be done.

## Notes

- **Precision.** `--precision bf16` (the default) is as in the Reactant version: bfloat16 weights
  and activations, `Float32` LayerNorm parameters and optimizer master weights, and a `Float32`
  loss. `tf32` runs `Float32` matrix multiplications on TF32 tensor cores
  (`CUDA.math_mode!(CUDA.FAST_MATH)`), and `f32` keeps them exact.
- **Attention.** With cuDNN.jl loaded, the causal `Attention` layers of the bfloat16 model run
  cuDNN's fused flash-attention kernels, forward and backward, so the `(T, T, heads, batch)`
  attention matrix is never materialized. In `Float32` they use NNlib's fused kernel.
- **Fused kernels.** LayerNorm (`NNlib.layernorm`) and the cross-entropy over the logits run as
  fused kernels, forward and backward, that accumulate in `Float32`.
- **Memory.** Zygote keeps every intermediate array of the forward pass for the backward pass, and
  CUDA.jl frees GPU memory only when Julia's garbage collector finalizes the arrays. `Flux.trainstep!`
  paces collections between micro-batches and runs a full collection after a step that compiles.
  The example needs no manual `GC.gc` calls. See the shared comparison for the allocator peaks.
- **Compilation.** There is no XLA compilation. The first step includes Julia compilation;
  package loading and data preparation happen before the step timer starts.
- **Automatic differentiation.** This version uses Zygote; the Reactant version uses Enzyme
  inside the compiled training step.
