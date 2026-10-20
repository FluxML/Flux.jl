# PyTorch port — GPT-2 pretraining

A PyTorch version of the Flux example in the parent directory ([`../train.jl`](../train.jl)),
adapted from Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt) (MIT license)
to match it: it tokenizes the data into the same rows as [`../data.jl`](../data.jl) (cached in
`../data/` on the first run), and has the same presets, model, initialization, optimizer,
schedule, evaluation and log format, so the two can be compared on the same GPU and the same
tokens.

Two choices follow Flux rather than build-nanogpt: gradients are clipped per parameter (Flux's
`ClipNorm`) instead of by their global norm, and `--attn math` selects materialized attention —
what Flux runs under Reactant — instead of flash attention, for a like-for-like comparison.

## Running

```console
$ cd examples/gpt2_fineweb/pytorch
$ uv sync                                     # torch (CUDA 13.0 wheels), tokenizers, datasets
$ uv run python train_gpt2.py --preset tinystories
$ uv run python train_gpt2.py --preset fineweb --attn math --steps 200 --eval-every 0 --hellaswag-every 0
```

`torch` comes from the CUDA 13.0 wheel index (`cu130`), which needs an NVIDIA driver supporting
CUDA 13; drop the `[tool.uv.sources]` / `[[tool.uv.index]]` blocks from `pyproject.toml` to use the
default wheels. The options mirror the Flux script's (`--batchsize`, `--steps`, `--lr`,
`--eval-every`, `--hellaswag-every`, `--precision {bf16,tf32,f32}`, `--peak-tflops`), plus `--attn`
and `--compile` (`torch.compile`).

## Performance comparison

GPT-2 124M (`--preset fineweb --batchsize 16`, gradient accumulation to 2^19 tokens per step) on an
**RTX 5090**, Julia 1.13 / Flux 0.16 / Reactant 0.2.295 vs PyTorch 2.14.1 (cu130), steady-state
steps:

| | tokens/s | s/step | TFLOPS | first step |
| --- | --- | --- | --- | --- |
| Flux + Reactant, `--precision bf16` | 111k | 4.7 | 95 | 723 s |
| PyTorch, bf16 autocast, `--attn math` | 67k | 7.8 | 58 | 8.0 s |
| PyTorch, bf16 autocast, `--attn flash` | 158k | 3.3 | 135 | 3.6 s |

Flux materializes the attention matrix, as Reactant has no fused attention for traced arrays yet
([Reactant.jl#3510](https://github.com/EnzymeAD/Reactant.jl/issues/3510)): with the same
materialized attention, PyTorch is 1.7× slower than Flux, but its default flash-attention kernel
makes it 1.4× faster. Flux's first step includes compiling the whole training step with XLA.
