# GPT-2 pretraining on FineWeb-Edu

Pretrains a **GPT-2** language model from scratch, following Karpathy's
[build-nanogpt](https://github.com/karpathy/build-nanogpt) reproduction of GPT-2 (124M): the same
architecture, data (the 10B-token sample of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)),
optimizer and schedule. The same training script comes in four versions, side by side:

| folder | framework | the training step |
| --- | --- | --- |
| [`flux-reactant/`](flux-reactant/) | Flux + [Reactant](https://github.com/EnzymeAD/Reactant.jl) | compiled into a single XLA program by `Flux.trainstep!`; no CUDA.jl. **The reference version.** |
| [`flux-cuda/`](flux-cuda/) | Flux + [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) | runs eagerly on `CuArray`s, differentiated by Zygote, with cuDNN's flash attention |
| [`lux-reactant-nitro/`](lux-reactant-nitro/) | Lux + [ReactantNitro](https://github.com/EnzymeAD/ReactantNitro.jl) | an XLA program compiled by Nitro, which runs the loop; differentiated by Enzyme |
| [`pytorch/`](pytorch/) | PyTorch | a port of the reference, for comparison |

The two Flux versions share everything but their entry point: [`flux-cuda/train.jl`](flux-cuda/train.jl)
includes the model, data pipeline, HellaSwag evaluation and training loop from
[`flux-reactant/`](flux-reactant/), and only chooses the device and the precision of matrix
multiplications. The Lux version includes the data pipeline of [`flux-reactant/`](flux-reactant/),
and writes the model, the optimizer and the evaluations as a Nitro experiment. The PyTorch port
reimplements them with the same presets, initialization, optimizer, schedule, evaluation and log
format. The four read the same tokenized data from `data/`, which whichever runs first writes.

## What it does

- **Data** — the documents are streamed from the Hub and tokenized with GPT-2's BPE (the
  `openai-community/gpt2` tokenizer), each prefixed with `<|endoftext|>`, and the token ids are
  packed into rows of `block_size + 1` tokens. On the first run, the rows are written to
  `data/<dataset>/` (2 bytes per token), which later runs load; `datasets` memory-maps them, and
  training reads them sequentially. In Julia, this goes through
  [HuggingFaceDatasets.jl](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl) and
  [HuggingFaceTokenizers.jl](https://github.com/MurrellGroup/HuggingFaceTokenizers.jl).
- **Model** — GPT-2: token and learned position embeddings, a stack of pre-LayerNorm transformer
  blocks (causal attention with 12 heads, its queries, keys and values computed by one projection,
  and a GELU MLP), a final LayerNorm, and an output layer tied to the token embedding. The
  vocabulary is padded from 50257 to 50304 tokens. Weights are initialized as in GPT-2 (std 0.02,
  residual projections scaled by `1/√(2·n_layer)`).
- **Training** — `AdamW` (β = (0.9, 0.95), weight decay 0.1 on matrices only), gradient clipping,
  linear warmup and cosine decay of the learning rate, and gradient accumulation to reach 2^19
  tokens per optimizer step. By default, weights and activations are in bfloat16, LayerNorm
  parameters stay in `Float32`, and the optimizer updates `Float32` copies of the parameters.
- **Evaluation** — validation loss, text samples from the prompt *"Hello, I'm a language model,"*
  (top-k sampling) and, for FineWeb-Edu, [HellaSwag](https://rowanzellers.com/hellaswag)
  accuracy, every 250 steps. The Julia scripts save checkpoints with the model and optimizer
  state to resume from.

Two presets:

| preset | model | data | tokens/step | steps | tokens |
| --- | --- | --- | --- | --- | --- |
| `fineweb` | GPT-2 124M (12 layers, 768 wide, context 1024) | FineWeb-Edu `sample-10BT` | 2^19 | 19073 | 10B |
| `tinystories` | 30M (6 layers, 384 wide, context 512) | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 2^16 | 7232 | 474M (1 epoch) |

## Running

```console
$ cd examples/gpt2_fineweb/flux-reactant      # or flux-cuda
$ julia --project=. -e 'using Pkg; Pkg.instantiate()'
$ julia --project=. train.jl --preset tinystories
$ julia --project=. train.jl --preset fineweb

$ cd examples/gpt2_fineweb/lux-reactant-nitro
$ julia --project=. -e 'using Pkg; Pkg.instantiate()'
$ julia --threads=1,0 --project=. train.jl --preset tinystories

$ cd examples/gpt2_fineweb/pytorch
$ uv sync
$ uv run python train_gpt2.py --preset fineweb
```

The Flux environments use Flux from this checkout and NNlib 0.9.59, which includes the Reactant
mixed precision fix of NNlib 0.9.56 ([NNlib.jl#872](https://github.com/FluxML/NNlib.jl/pull/872)).
The Lux environment uses registered releases only (Lux 1.31, ReactantNitro 0.2, Reactant 0.2.307).

The scripts share the preset, batch size, precision, learning-rate and evaluation options
(`--help` lists them), write `log/log.txt` in their own folder, and print the same training log
lines. The Julia versions also save resumable checkpoints. The first run with a preset
streams and tokenizes its dataset into `data/<dataset>/`: ~1 min and 0.95 GB for TinyStories,
~30 min (28.5 GB streamed) and 20 GB for FineWeb-Edu. See each folder's README for the details.

## Performance

Measured on **28 October 2026** on an **RTX 5090 (32 GB)** with driver 595.84, except for
Lux + ReactantNitro, measured again with flash attention on **6 November 2026** (Reactant 0.2.307),
and the two Flux versions, measured again on **12 November 2026** with driver 595.99.
GPT-2 124M uses
`--preset fineweb --batchsize 16 --precision bf16`: 32 micro-batches of 16 × 1024 tokens per
optimizer step (2^19 tokens). Each run takes **eight optimizer steps**, with validation,
sampling, HellaSwag and checkpoints disabled. Steady timings and rates use steps 2–8; the first
step includes compilation, but excludes package loading and data preparation. These are short
benchmarks, not full training runs.

Both Flux environments use Julia 1.13.2 (32 threads), Flux master at
[`520788b`](https://github.com/FluxML/Flux.jl/commit/520788b44946473fcf27ca32f7899bd757591527),
which includes [#2825](https://github.com/FluxML/Flux.jl/pull/2825), NNlib 0.9.59,
Optimisers 0.4.11 and MLDataDevices 1.18. The CUDA.jl one uses CUDA.jl and cuDNN.jl 6.5.1,
GPUArrays 12.2.0, cuDNN 9.28 and Zygote 0.7.18; the Reactant one uses Reactant 0.2.308.
The Lux environment uses Julia 1.13.2 with `--threads=1,0`, Lux 1.31.5, ReactantNitro 0.2.2,
Optimisers 0.4.11, NNlib 0.9.58 and Reactant 0.2.307.
PyTorch is 2.14.1+cu130, with cuDNN 9.24, `datasets` 5.1 and 16 CPU threads. Each compiled
PyTorch run starts with separate empty Inductor and Triton caches.

| | tokens/s | s/step | TFLOPS | first step |
| --- | --- | --- | --- | --- |
| [Flux + Reactant](flux-reactant/) | 186k | 2.82 | 158.9 | 399 s |
| [Lux + ReactantNitro](lux-reactant-nitro/) | 196k | 2.68 | 167.3 | 106.8 s |
| [Flux + CUDA.jl](flux-cuda/) | 171k | 3.07 | 146.2 | 45.7 s |
| [PyTorch](pytorch/), `--attn math` | 67k | 7.78 | 57.7 | 8.0 s |
| [PyTorch](pytorch/), `--attn math --compile` | 107k | 4.90 | 91.6 | 18.6 s |
| [PyTorch](pytorch/), `--attn flash` (the default) | 158k | 3.31 | 135.5 | 3.5 s |
| [PyTorch](pytorch/), `--attn flash --compile` | 163k | 3.21 | 139.5 | 12.7 s |

Allocator peaks from the **same eight-step runs**, including model/optimizer setup and the
first step. GiB means 2^30 bytes. These are allocator counters, not total device memory as
reported by `nvidia-smi`.

| | peak memory in use | peak reserved |
| --- | --- | --- |
| Flux + Reactant | 9.9 GiB | 16.0 GiB |
| Lux + ReactantNitro | 11.3 GiB | 16.0 GiB |
| Flux + CUDA.jl | 19.2 GiB | 19.3 GiB |
| PyTorch, `--attn math` | 26.4 GiB | 27.6 GiB |
| PyTorch, `--attn math --compile` | 25.5 GiB | 25.9 GiB |
| PyTorch, `--attn flash` | 16.5 GiB | 16.7 GiB |
| PyTorch, `--attn flash --compile` | 16.5 GiB | 16.8 GiB |

Reactant preallocation was disabled with `XLA_REACTANT_GPU_PREALLOCATE=false`; the scripts
otherwise reserve 90% of GPU memory. Its counters are `Reactant.XLA.allocatorstats()`'s
`peak_bytes_in_use` and `peak_pool_bytes`. CUDA uses the memory pool's `USED_MEM_HIGH` and
`RESERVED_MEM_HIGH`; PyTorch uses `max_memory_allocated()` and `max_memory_reserved()`.
A second run of the Lux version gave the same steady speed (within 0.3%) and peak in use, and a
first step of 101 s. Both Reactant versions reserve 16.0 GiB because the host runs ahead of the
device: Flux + Reactant reserves 12.0 GiB when it reads each loss back as soon as its micro-batch
is queued (see below).

To reproduce the Julia timing run from either Flux environment:

```console
$ XLA_REACTANT_GPU_PREALLOCATE=false julia --threads=32 --project=. train.jl --preset fineweb --steps 8 --eval-every 0 --hellaswag-every 0 --ckpt-every 0
```

and from the Lux environment:

```console
$ XLA_REACTANT_GPU_PREALLOCATE=false julia --threads=1,0 --project=. train.jl --preset fineweb --steps 8 --eval-every 0 --hellaswag-every 0 --ckpt-every 0
```

For PyTorch, use `--preset fineweb --steps 8 --eval-every 0 --hellaswag-every 0` and each
attention/compilation combination in the table. Set `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` to fresh directories for each compiled run, and `OMP_NUM_THREADS=16` and
`MKL_NUM_THREADS=16` to match the CPU thread count.

- **Reactant** compiles the forward pass, the backward pass and the optimizer update into one XLA
  program, which fuses the elementwise operations and plans the memory of the whole step. Since
  Reactant 0.2.305, NNlib's `scaled_dot_product_attention`, which the attention calls,
  compiles to cuDNN's flash-attention kernels for bfloat16 inputs, in the forward and the backward
  pass ([Reactant.jl#3615](https://github.com/EnzymeAD/Reactant.jl/pull/3615)). With Reactant
  0.2.301, which materialized the attention matrix, Flux trained at 131k tokens/s with 18.9 GiB
  in use: it is now 1.42× faster with half the memory, 9% faster than CUDA.jl and 14% faster than
  compiled PyTorch with flash attention. Its first step includes compiling that program, which
  took 636 s with Reactant 0.2.301, and 787 s with Reactant 0.2.307 when the model computed the
  queries, keys and values with three projections (Flux's `Attention`) instead of one. The compile
  time grows quadratically with the number of arrays
  ([Reactant.jl#3511](https://github.com/EnzymeAD/Reactant.jl/issues/3511)), and there were 196
  parameter arrays instead of 148, each with its optimizer state. Since NNlib 0.9.56, the activations stay in
  bfloat16 with `Float32` LayerNorm parameters, fixing the promotion in NNlib 0.9.55.
- **Lux + ReactantNitro** compiles the same program through Nitro, which owns the loop,
  accumulating the gradients of the 32 micro-batches and updating the parameters with the
  optimizer. Its model calls `scaled_dot_product_attention` with `is_causal = true` too, and
  reaches 196k tokens/s, 5% above Flux + Reactant, compared with 141k tokens/s and 20.1 GiB in
  use with the attention matrix materialized (NNlib's deprecated `dot_product_attention` with a
  causal mask, on the same Reactant). Its first step takes 107 s, compared with 399 s; why has not
  been investigated, but Nitro flattens the parameters and the optimizer state into one buffer per
  parameter group, which keeps the optimizer program from growing with the number of arrays
  ([Reactant.jl#3511](https://github.com/EnzymeAD/Reactant.jl/issues/3511)). Its precision policy
  is PyTorch's: `Float32` parameters, of which the model sees a bfloat16 copy, but for the
  LayerNorm ones, with LayerNorm and the loss computed in `Float32`. The 30M-parameter
  `tinystories` run trains to a validation loss close to the Flux version's (1.501 after 7000
  steps, against 1.488 after 7232), see [`lux-reactant-nitro/`](lux-reactant-nitro/#results).
- **Flux + Reactant against Lux + ReactantNitro.** Flux's compiled forward and backward passes,
  accumulating the gradient, take 82.6 ms per micro-batch, against 83.8 ms for a 32nd of Lux's
  whole step: the 0.14 s per step that Flux loses are spent around them. Under tracing,
  `AccumGrad` cannot skip the rules after it, so Flux's step runs the whole optimizer on every
  micro-batch and selects the old state back on the 31 that only accumulate: 4.9 ms each, where
  Nitro runs a separate optimizer program once per step, 0.11 s per step in all. On synthetic
  batches, a step that only accumulates on those micro-batches runs at Lux's 196k tokens/s. Since
  [#2825](https://github.com/FluxML/Flux.jl/pull/2825), `trainstep!` returns the loss on the
  device, and the training loop reads the 32 losses once every micro-batch of the step is queued,
  so that loading a batch overlaps the micro-batches before it, as Nitro does. Reading each loss
  back as soon as its micro-batch is queued, as `trainstep!` did before
  ([#2801](https://github.com/FluxML/Flux.jl/issues/2801)), the same run takes 2.97 s per step
  (176k tokens/s), as fast as with Reactant 0.2.307 and NNlib master on 6 November.
- **CUDA.jl** runs the attention as cuDNN's flash-attention kernels, LayerNorm and the
  cross-entropy as fused kernels. Flux 0.16.15 paces garbage collections between micro-batches
  and runs a full collection after the first step that compiles; no manual collections are needed.
- **PyTorch** runs eagerly too, with flash attention by default; `--attn math` materializes the
  attention matrix. `--compile` compiles the model, excluding the
  loss and optimizer update. Its precision policy differs from Flux's: it keeps `Float32`
  parameters, uses bfloat16 autocast, and casts logits to `Float32` for the loss. Flux uses
  bfloat16 weights with `Float32` LayerNorm parameters and master weights, and accumulates the
  loss in `Float32` without copying all the logits to that type. See [`pytorch/`](pytorch/).
