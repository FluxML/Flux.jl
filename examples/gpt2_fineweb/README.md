# GPT-2 pretraining on FineWeb-Edu (Flux + Reactant + HuggingFaceDatasets.jl)

Pretrains a **GPT-2** language model from scratch, following Karpathy's
[build-nanogpt](https://github.com/karpathy/build-nanogpt) reproduction of GPT-2 (124M): the same
architecture, data (the 10B-token sample of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)),
optimizer and schedule. The training step is compiled to a single XLA program with
[Reactant](https://github.com/EnzymeAD/Reactant.jl) through `Flux.trainstep!`, so no CUDA.jl is
needed; the data comes from the HuggingFace Hub through
[HuggingFaceDatasets.jl](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl) and is tokenized with
[HuggingFaceTokenizers.jl](https://github.com/MurrellGroup/HuggingFaceTokenizers.jl).

| file | what |
| --- | --- |
| [`data.jl`](data.jl) | streams a dataset from the Hub and tokenizes it into flat `UInt16` token files; the batch loader |
| [`model.jl`](model.jl) | the GPT-2 model, the loss, and sampling |
| [`train.jl`](train.jl) | the training script: presets, gradient accumulation, evaluation, checkpoints |
| [`hellaswag.jl`](hellaswag.jl) | [HellaSwag](https://rowanzellers.com/hellaswag) accuracy, during training or for a saved checkpoint |
| [`Project.toml`](Project.toml) | the example's environment |
| [`pytorch/`](pytorch/) | a PyTorch port of the same script, for a Flux-vs-PyTorch comparison |

## What it does

- **Data** — the documents are streamed from the Hub and tokenized with GPT-2's BPE (the
  `openai-community/gpt2` tokenizer), each prefixed with `<|endoftext|>`. The token ids are written
  to `data/<dataset>/{train,val}.bin` (2 bytes per token) and memory-mapped during training, which
  reads them sequentially in windows of `block_size + 1` tokens.
- **Model** — GPT-2: token and learned position embeddings, a stack of pre-LayerNorm transformer
  blocks (causal `Attention` with 12 heads, GELU MLP), a final LayerNorm, and an output layer
  tied to the token embedding. The vocabulary is padded from 50257 to 50304 tokens. Weights are
  initialized as in GPT-2 (std 0.02, residual projections scaled by `1/√(2·n_layer)`).
- **Training** — `AdamW` (β = (0.9, 0.95), weight decay 0.1 on matrices only), gradient clipping,
  linear warmup and cosine decay of the learning rate, and gradient accumulation with
  `AccumGrad` to reach 2^19 tokens per optimizer step. Each micro-batch is one `Flux.trainstep!`;
  the first call compiles the whole step (forward, backward and optimizer update).
- **Evaluation** — validation loss, text samples from the prompt *"Hello, I'm a language model,"*
  and, for FineWeb-Edu, HellaSwag accuracy, every 250 steps; checkpoints with the model and
  optimizer state to resume from.

Two presets:

| preset | model | data | tokens/step | steps | tokens |
| --- | --- | --- | --- | --- | --- |
| `fineweb` | GPT-2 124M (12 layers, 768 wide, context 1024) | FineWeb-Edu `sample-10BT` | 2^19 | 19073 | 10B |
| `tinystories` | 30M (6 layers, 384 wide, context 512) | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 2^16 | 7232 | 474M (1 epoch) |

## Running

```console
$ cd examples/gpt2_fineweb
$ julia --project=. -e 'using Pkg; Pkg.instantiate()'

$ julia --project=. data.jl tinystories             # ~1 min, 0.95 GB
$ julia --project=. train.jl --preset tinystories   # ~30 min on an RTX 5090

$ julia --project=. data.jl fineweb                 # ~25 min, 28.5 GB download, 20 GB of tokens
$ julia --project=. train.jl --preset fineweb
$ julia --project=. train.jl --preset fineweb --resume log/checkpoint_05000.jld2
$ julia --project=. hellaswag.jl log/checkpoint_19073.jld2
```

Options (`--help` lists them all): `--batchsize` is the number of sequences per micro-batch (the
preset's tokens per step are reached by accumulating gradients, so it only trades speed for
memory), `--steps`, `--lr`, `--eval-every`, `--hellaswag-every`, `--ckpt-every` (0 disables
each), `--precision` (see below), `--resume`, `--seed`, `--data-dir`, `--log-dir`, and
`--peak-tflops` to report the model FLOPs utilization. Every line printed is also appended to
`log/log.txt`.

## Results

Measured on an **RTX 5090 (32 GB)**, Julia 1.13, Flux 0.16, Reactant 0.2.295.

**`tinystories`** — 22 minutes of training (0.18 s per step, ~365k tokens/s) after ~3 minutes of
compilation. The validation loss goes 3.21 → 2.01 → 1.67 → **1.488** at steps 250, 1000, 3000 and
7232 (1.486 with `Float32` parameters, which took 29 minutes), and the samples are fluent:

> Hello, I'm a language model, Ben. You can talk or sing or laugh. That is nice. Can you teach
> me?" Anna said, "Yes, I can. I can…

**`fineweb`** — 111k tokens/s (~95 TFLOPS) at the default `--batchsize 16`, i.e. 4.7 s per step of
2^19 tokens and ~25 hours for the full run, after ~12 minutes of compilation; with `Float32`
parameters (and bfloat16 matrix multiplications) it is 1.7× slower. PyTorch's port runs at 158k
tokens/s with flash attention, and at 67k with the same materialized attention as Flux — see
[`pytorch/`](pytorch/).

The HellaSwag evaluation reproduces build-nanogpt's numbers for OpenAI's GPT-2 (124M) weights
loaded into this model: acc = 0.2861, acc_norm = 0.2956 (reference: 0.2859, 0.2955).

## Notes

- **Precision.** With `--precision bf16` (the default) the model is converted with `Flux.bf16`,
  so parameters and activations are bfloat16, while the optimizer keeps and updates `Float32`
  copies of the parameters (`MixedPrecision(Float32, rule)`), and the loss is computed in
  `Float32`. `tf32` keeps everything in `Float32` with XLA's default matmul precision on NVIDIA
  GPUs (TF32-like), and `f32` is exact (`Reactant.with_config(; dot_general_precision =
  PrecisionConfig.HIGHEST)` around the training loop, as the setting is read when a step is
  compiled).
- **Attention.** Under Reactant, attention goes through NNlib's standard implementation, which
  materializes the `(T, T, heads, batch)` attention matrix: there is no fused (flash) attention
  for traced arrays yet ([Reactant.jl#3510](https://github.com/EnzymeAD/Reactant.jl/issues/3510)).
  This bounds the micro-batch size and costs speed; the example uses `Attention(...; is_causal=true)`,
  so it will benefit from a fused implementation without changes.
- **Compilation.** The first training step takes ~3 minutes for `tinystories` and ~12 for `fineweb`
  on the RTX 5090, but MLIR and XLA account for only about a minute of it. The rest is Julia
  compiling Reactant's tracing code and the function that calls the compiled step, whose compile
  time grows quadratically with the number of parameter and optimizer-state arrays
  ([Reactant.jl#3511](https://github.com/EnzymeAD/Reactant.jl/issues/3511)). The validation loss
  and HellaSwag scoring are compiled once each too.
- **Memory.** XLA preallocates a fraction of the GPU memory; the script raises it from 0.75 to 0.9
  (`XLA_REACTANT_GPU_MEM_FRACTION`).
- **Gradient clipping** uses `ClipNorm(1.0)`, which clips each parameter array separately rather than
  the global norm as build-nanogpt does. The PyTorch port clips per parameter too.
- **Python.** HuggingFaceDatasets.jl and HuggingFaceTokenizers.jl run the Python `datasets` and
  `tokenizers` packages, installed automatically by CondaPkg in `.CondaPkg/`. The text stays in
  Python (`with_format(nothing)`) and only the token ids are converted to Julia.
