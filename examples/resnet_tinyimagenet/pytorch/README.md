# PyTorch port — ResNet-18 on Tiny-ImageNet-200

A line-for-line PyTorch translation of the Flux example in the parent directory
([`../resnet_tinyimagenet.jl`](../resnet_tinyimagenet.jl)), kept deliberately close so the two
frameworks can be compared head to head on the same GPU and the same data.

Same everything: small-image ResNet-18 (3×3 stride-1 stem, four 2-block stages 64→128→256→512,
global average pool, linear head, **11.27M params**), the same `zh-plus/tiny-imagenet` dataset
pulled through HuggingFace `datasets`, the same ImageNet normalization, the same pad-4 random-crop +
horizontal-flip augmentation, and `AdamW(lr=1e-3)` with **`weight_decay=0`** (matching Flux's
`AdamW` default `lambda=0` — PyTorch otherwise defaults to `0.01`) and cross-entropy loss.

## Running

```console
$ cd examples/resnet_tinyimagenet/pytorch
$ uv sync                                              # torch (cu128) + datasets
$ uv run python resnet_tinyimagenet.py                 # 30 epochs (the default)
$ uv run python resnet_tinyimagenet.py --epochs 5
$ uv run python resnet_tinyimagenet.py --benchmark-epochs 3   # timed short run
$ uv run python resnet_tinyimagenet.py --help
```

The `[project]` `torch` dependency is pinned to the CUDA 12.8 wheel index (`cu128`) so it runs on
recent NVIDIA GPUs (Blackwell / RTX 50-series included). Drop the `[tool.uv.sources]` /
`[[tool.uv.index]]` blocks from `pyproject.toml` to fall back to the default CPU/CUDA wheels.

Each epoch prints its pure training wall-time and throughput (`[train]`), followed by train/val
loss and top-1 accuracy (`[eval]`).

## Performance comparison

Measured on this workstation — **RTX 5090 (32 GB)**, Julia 1.12.6 / Flux 0.16 / CUDA.jl vs
PyTorch 2.11.0+cu128 — same model (11.27M params), batch size 128, 4 data-loading workers,
`AdamW(1e-3)`. Throughput is the pure training loop (forward + backward + step) over the 100k-image
train set; `flux_bench.jl` in this directory reproduces the Flux side with the same
per-epoch timing.

| metric | Flux | PyTorch |
| --- | --- | --- |
| params | 11.27M | 11.27M |
| **steady-state train** | **~33.0 s/epoch · ~3010 img/s** | **~30.3 s/epoch · ~3280 img/s** |
| first train epoch | 70.2 s (incl. compilation) | 30.5 s |
| epoch-0 full eval (110k imgs) | 65 s cold → 10.4 s warm | ~few s |
| GPU during train | 90–97 % · ~540 W | 100 % · ~545 W |
| GPU memory | pool reserves ~32 GB¹ | ~3.8 GB |
| process warmup before epoch 1 | package load + precompile (~min) | ~seconds |

¹ CUDA.jl's memory pool *reserves* (and retains) freed device memory rather than returning it to
the driver; live working-set usage is a small fraction of that. PyTorch's caching allocator holds
only what it needs here (~3.8 GB).

**Takeaways**

- **Steady-state throughput is within ~8 %** — both frameworks are GPU-compute-bound (both pin the
  card at ~540 W, i.e. tensor-core saturated), so this small ResNet-18 runs at essentially the same
  speed once warm. PyTorch is marginally ahead (cuDNN autotuning via `benchmark=True`, plus its
  vectorized-on-GPU augmentation vs. Flux's per-image CPU augmentation).
- **The real difference is warmup, not steady speed.** Julia pays a one-time first-call
  compilation cost: epoch 1 is ~2× the steady time and the first full-train eval is ~6× the warm
  eval, on top of a minute-plus of package load/precompile at startup. PyTorch reaches steady speed
  on epoch 1. For a 30-epoch run this amortizes to a few percent; for short runs it dominates.
- **Accuracy trajectories are comparable** (run-to-run variance from init/augmentation RNG); both
  climb steadily toward the ~50 % top-1 the README quotes for a full 30-epoch run.

## Notes on faithfulness

- **Augmentation** is vectorized over the batch (one crop offset + flip decision per batch) rather
  than per-image as in the Julia `augment`. This is cheaper and, at batch size 128, statistically
  negligible for this comparison; a per-image variant is straightforward if exactness is wanted.
- **Data layout**: HuggingFace `datasets` with `.with_format("torch")` hands an Image column back
  as an NCHW uint8 tensor already, so no permute is needed (the Julia "julia" format instead yields
  CWHN and permutes to WHCN).
- Normalization and augmentation run on the GPU, matching where the Flux version does the same work.
