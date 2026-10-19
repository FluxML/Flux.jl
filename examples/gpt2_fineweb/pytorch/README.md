# PyTorch port — GPT-2 pretraining

A PyTorch version of the Flux example in the parent directory ([`../train.jl`](../train.jl)),
adapted from Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt) (MIT license)
to match it: it reads the token files written by [`../data.jl`](../data.jl), and has the same
presets, model, initialization, optimizer, schedule, evaluation and log format, so the two can be
compared on the same GPU and the same tokens.

Two choices follow Flux rather than build-nanogpt: gradients are clipped per parameter (Flux's
`ClipNorm`) instead of by their global norm, and `--attn math` selects materialized attention —
what Flux runs under Reactant — instead of flash attention, for a like-for-like comparison.

## Running

```console
$ cd examples/gpt2_fineweb
$ julia --project=. data.jl tinystories      # the token files are shared with the Flux script
$ cd pytorch
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

| | tokens/s | s/step | TFLOPS |
| --- | --- | --- | --- |
| Flux + Reactant, `Float32` parameters, bf16 matmuls | 64k | 8.2 | 55 |
| PyTorch, bf16 autocast, `--attn math` | 67k | 7.8 | 58 |
| PyTorch, bf16 autocast, `--attn flash` | 158k | 3.3 | 135 |

With the same materialized attention the two are within 5%; the flash-attention kernel PyTorch
uses by default is 2.4× faster. Reactant has no fused attention for traced arrays yet
([Reactant.jl#3510](https://github.com/EnzymeAD/Reactant.jl/issues/3510)).
