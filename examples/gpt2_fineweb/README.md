# GPT-2 pretraining on FineWeb-Edu

Pretrains a **GPT-2** language model from scratch, following Karpathy's
[build-nanogpt](https://github.com/karpathy/build-nanogpt) reproduction of GPT-2 (124M): the same
architecture, data (the 10B-token sample of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)),
optimizer and schedule. The same training script comes in three versions, side by side:

| folder | framework | the training step |
| --- | --- | --- |
| [`flux-reactant/`](flux-reactant/) | Flux + [Reactant](https://github.com/EnzymeAD/Reactant.jl) | compiled into a single XLA program by `Flux.trainstep!`; no CUDA.jl. **The reference version.** |
| [`flux-cuda/`](flux-cuda/) | Flux + [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) | runs eagerly on `CuArray`s, differentiated by Zygote, with cuDNN's flash attention |
| [`pytorch/`](pytorch/) | PyTorch | a port of the reference, for comparison |

The two Flux versions share everything but their entry point: [`flux-cuda/train.jl`](flux-cuda/train.jl)
includes the model, data pipeline, HellaSwag evaluation and training loop from
[`flux-reactant/`](flux-reactant/), and only chooses the device and the precision of matrix
multiplications. The PyTorch port reimplements them with the same presets, initialization,
optimizer, schedule, evaluation and log format. The three read the same tokenized data from
`data/`, which whichever runs first writes.

## What it does

- **Data** — the documents are streamed from the Hub and tokenized with GPT-2's BPE (the
  `openai-community/gpt2` tokenizer), each prefixed with `<|endoftext|>`, and the token ids are
  packed into rows of `block_size + 1` tokens. On the first run, the rows are written to
  `data/<dataset>/` (2 bytes per token), which later runs load; `datasets` memory-maps them, and
  training reads them sequentially. In Julia, this goes through
  [HuggingFaceDatasets.jl](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl) and
  [HuggingFaceTokenizers.jl](https://github.com/MurrellGroup/HuggingFaceTokenizers.jl).
- **Model** — GPT-2: token and learned position embeddings, a stack of pre-LayerNorm transformer
  blocks (causal attention with 12 heads, GELU MLP), a final LayerNorm, and an output layer
  tied to the token embedding. The vocabulary is padded from 50257 to 50304 tokens. Weights are
  initialized as in GPT-2 (std 0.02, residual projections scaled by `1/√(2·n_layer)`).
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

$ cd examples/gpt2_fineweb/pytorch
$ uv sync
$ uv run python train_gpt2.py --preset fineweb
```

The Julia environments use Flux from this checkout and NNlib from GitHub `master`. The latter
includes the Reactant mixed precision fix in [NNlib.jl#872](https://github.com/FluxML/NNlib.jl/pull/872),
which is not in the registered NNlib 0.9.55 release.

The scripts share the preset, batch size, precision, learning-rate and evaluation options
(`--help` lists them), write `log/log.txt` in their own folder, and print the same training log
lines. The Julia versions also save resumable checkpoints. The first run with a preset
streams and tokenizes its dataset into `data/<dataset>/`: ~1 min and 0.95 GB for TinyStories,
~30 min (28.5 GB streamed) and 20 GB for FineWeb-Edu. See each folder's README for the details.

## Performance

Measured on **28 October 2026**, on an **RTX 5090 (32 GB)**, driver 595.84. GPT-2 124M uses
`--preset fineweb --batchsize 16 --precision bf16`: 32 micro-batches of 16 × 1024 tokens per
optimizer step (2^19 tokens). Each run takes **eight optimizer steps**, with validation,
sampling, HellaSwag and checkpoints disabled. Steady timings and rates use steps 2–8; the first
step includes compilation, but excludes package loading and data preparation. These are short
benchmarks, not full training runs.

Both Julia environments use Julia 1.13.2 (32 threads), Flux 0.16.15, Optimisers 0.4.11,
MLDataDevices 1.18 and NNlib master at
[`8790de4`](https://github.com/FluxML/NNlib.jl/commit/8790de4ba6e5c4bf48f5392888dd8ceb6a799ca0).
CUDA.jl and cuDNN.jl are 6.5.0, with cuDNN 9.27 and Zygote 0.7.15; Reactant is 0.2.301.
PyTorch is 2.14.1+cu130, with cuDNN 9.24, `datasets` 5.1 and 16 CPU threads. Each compiled
PyTorch run starts with separate empty Inductor and Triton caches.

| | tokens/s | s/step | TFLOPS | first step |
| --- | --- | --- | --- | --- |
| [Flux + Reactant](flux-reactant/) | 131k | 4.01 | 111.7 | 636 s |
| [Flux + CUDA.jl](flux-cuda/) | 167k | 3.13 | 143.1 | 53.1 s |
| [PyTorch](pytorch/), `--attn math` | 67k | 7.78 | 57.7 | 8.0 s |
| [PyTorch](pytorch/), `--attn math --compile` | 107k | 4.90 | 91.6 | 18.6 s |
| [PyTorch](pytorch/), `--attn flash` (the default) | 158k | 3.31 | 135.5 | 3.5 s |
| [PyTorch](pytorch/), `--attn flash --compile` | 163k | 3.21 | 139.5 | 12.7 s |

Allocator peaks from the **same eight-step runs**, including model/optimizer setup and the
first step. GiB means 2^30 bytes. These are allocator counters, not total device memory as
reported by `nvidia-smi`.

| | peak memory in use | peak reserved |
| --- | --- | --- |
| Flux + Reactant | 18.9 GiB | 28.2 GiB |
| Flux + CUDA.jl | 18.7 GiB | 18.7 GiB |
| PyTorch, `--attn math` | 26.4 GiB | 27.6 GiB |
| PyTorch, `--attn math --compile` | 25.5 GiB | 25.9 GiB |
| PyTorch, `--attn flash` | 16.5 GiB | 16.7 GiB |
| PyTorch, `--attn flash --compile` | 16.5 GiB | 16.8 GiB |

Reactant preallocation was disabled with `XLA_REACTANT_GPU_PREALLOCATE=false`; the script
otherwise reserves 90% of GPU memory. Its counters are `Reactant.XLA.allocatorstats()`'s
`peak_bytes_in_use` and `peak_pool_bytes`. CUDA uses the memory pool's `USED_MEM_HIGH` and
`RESERVED_MEM_HIGH`; PyTorch uses `max_memory_allocated()` and `max_memory_reserved()`.

To reproduce the Julia timing run from either environment:

```console
$ XLA_REACTANT_GPU_PREALLOCATE=false julia --threads=32 --project=. train.jl --preset fineweb --steps 8 --eval-every 0 --hellaswag-every 0 --ckpt-every 0
```

For PyTorch, use `--preset fineweb --steps 8 --eval-every 0 --hellaswag-every 0` and each
attention/compilation combination in the table. Set `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` to fresh directories for each compiled run, and `OMP_NUM_THREADS=16` and
`MKL_NUM_THREADS=16` to match the CPU thread count.

- **Reactant** compiles the forward pass, the backward pass and the optimizer update into one XLA
  program, which fuses the elementwise operations and plans the memory of the whole step, but it
  has no fused attention yet: it materializes the attention matrix
  ([Reactant.jl#3510](https://github.com/EnzymeAD/Reactant.jl/issues/3510)). Its first step
  includes compiling that program. NNlib master keeps the activations in bfloat16 with
  `Float32` LayerNorm parameters, fixing the promotion in NNlib 0.9.55.
- **CUDA.jl** runs the attention as cuDNN's flash-attention kernels, LayerNorm and the
  cross-entropy as fused kernels. Flux 0.16.15 paces garbage collections between micro-batches
  and runs a full collection after the first step that compiles; no manual collections are needed.
- **PyTorch** runs eagerly too, with flash attention by default; `--attn math` materializes the
  attention matrix as Flux does under Reactant. `--compile` compiles the model, excluding the
  loss and optimizer update. Its precision policy differs from Flux's: it keeps `Float32`
  parameters, uses bfloat16 autocast, and casts logits to `Float32` for the loss. Flux uses
  bfloat16 weights with `Float32` LayerNorm parameters and master weights, and accumulates the
  loss in `Float32` without copying all the logits to that type. See [`pytorch/`](pytorch/).
