# PyTorch port — GPT-2 pretraining

A PyTorch version of the Flux [GPT-2 example](../README.md)
([`../flux-reactant/train.jl`](../flux-reactant/train.jl)), adapted from Karpathy's
[build-nanogpt](https://github.com/karpathy/build-nanogpt) (MIT license) to match it: it tokenizes
the data as [`../flux-reactant/data.jl`](../flux-reactant/data.jl) does, into the same cached
dataset in `../data/` (whichever script runs first writes it, the others load it), and has the
same presets, model, initialization, optimizer, schedule, evaluation and log format, so they can
be compared on the same GPU and the same tokens.

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

Both ports require `datasets` 5.1 or later: earlier versions can skip the wrong documents in
the FineWeb-Edu stream, overlapping the training and validation splits.

## Performance comparison

The [shared comparison](../README.md#performance) records timings and allocator memory peaks
for GPT-2 124M on the RTX 5090, using eight optimizer steps with micro-batches of 16 and gradient
accumulation to 2^19 tokens per step.

`--attn math` materializes the attention matrix, as Flux does under Reactant
([Reactant.jl#3510](https://github.com/EnzymeAD/Reactant.jl/issues/3510)); `--attn flash` uses fused
attention, as Flux does on CUDA.jl. PyTorch keeps the model parameters in `Float32` and uses
bfloat16 autocast. Flux uses bfloat16 weights with `Float32` LayerNorm parameters and master
weights, so the precision policies differ.

`--compile` compiles the model, not the loss or optimizer update. Each compiled benchmark uses
its own empty `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` so the first step includes
compilation for the selected attention backend.
