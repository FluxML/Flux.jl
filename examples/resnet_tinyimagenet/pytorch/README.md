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

A normal run prints each epoch's training wall-time and throughput (`[train]`) followed by train/val
loss and top-1 accuracy (`[eval]`); `--benchmark-epochs N` times training only — no evaluation — to
line up with [`flux_bench.jl`](flux_bench.jl).

## Performance comparison

Measured on this workstation — **RTX 5090 (32 GB)**, Julia 1.12.6 / Flux 0.16.10 / **Reactant
0.2.278** (XLA) vs PyTorch 2.11.0+cu128 — same model (11.27M params), batch size 128, 4
data-loading workers, `AdamW(1e-3)`. Throughput is the pure training loop (forward + backward +
step) over the 100k-image train set, **timing only, no evaluation**: [`flux_bench.jl`](flux_bench.jl)
and `resnet_tinyimagenet.py --benchmark-epochs 3` are the matched drivers.

| metric | Flux (Reactant) | PyTorch |
| --- | --- | --- |
| params | 11.27M | 11.27M |
| **steady-state train** | **~32.5 s/epoch · ~3060 img/s** | **~30.1 s/epoch · ~3320 img/s** |
| first train epoch | **235.6 s** (incl. XLA compilation) | 40.0 s (incl. cuDNN autotune) |
| GPU utilization (steady) | ~96 % | ~100 % |
| GPU memory | **~16.8 GB** working set · 23.5 GB reserved by default¹ | ~3.8 GB |

¹ By default Reactant/XLA's BFC allocator *preallocates* a fixed fraction of the card
(`XLA_REACTANT_GPU_MEM_FRACTION`, default 0.75 → ~23.5 GB of the 32 GB here) on first use, regardless
of the model's needs — the strategy JAX uses too. Setting `XLA_REACTANT_GPU_PREALLOCATE=false`
allocates on demand and reveals the true footprint: **~16.8 GB, still ~4× PyTorch's ~3.8 GB**
(throughput is unchanged). So the gap is real, not just a reservation policy: the fused whole-step
executable appears to keep all forward activations live for the Enzyme reverse pass (plus gradient
shadows and cuDNN conv workspaces), whereas PyTorch's eager autograd frees activations more
incrementally. See Reactant's [GPU config](https://enzymead.github.io/Reactant.jl/dev/api/config#GPU-Configuration).

**Takeaways**

- **Steady-state throughput is within ~8 %** — both frameworks are GPU-compute-bound on this small
  ResNet-18, so once warm they run at essentially the same speed. PyTorch is marginally ahead (cuDNN
  autotuning via `benchmark=True`, plus its vectorized-on-GPU augmentation vs. Flux's per-image CPU
  augmentation). Reactant's steady-state matches the eager CUDA.jl path it replaced (~32.4 s/epoch).
- **The cost is warmup.** On the first `trainstep!`, Reactant compiles the whole training step
  (forward + Enzyme reverse + optimiser update) into a single XLA executable, so **epoch 1 is ~7× the
  steady time** (~236 s vs ~32 s) — on top of the minute-plus of package load/precompile at startup.
  PyTorch reaches steady speed on epoch 1 (its ~40 s is just cuDNN autotuning). The executable is
  cached and reused for every later step, so the compile is a one-time cost: a modest fraction of a
  30-epoch run, but dominant for short ones.
- **Accuracy trajectories are comparable** (run-to-run variance from init/augmentation RNG); both
  climb steadily toward the ~50 % top-1 the parent README quotes for a full 30-epoch run.

### Why the training loop moves batches to the device

Both the example and `flux_bench.jl` wrap the training loader in an
[MLDataDevices](https://github.com/LuxDL/MLDataDevices.jl) device iterator — `device(train_loader)` —
so every batch arrives on the accelerator (and the previous one is freed). On Reactant this is
**required**, not just an optimisation: `Flux.trainstep!` runs the compiled XLA step, which only
accepts device-resident arrays — a host-resident batch raises an error. The loop is one fused,
cached executable per step:

```julia
for (x, y) in train_loader          # x, y already on the Reactant device
    Flux.trainstep!(loss_fn, model, (x, y), opt_state)   # compiled once, reused every step/epoch
end
```

`trainstep!` differentiates `loss_fn` with Enzyme and folds the AdamW update into the compiled step,
returning the loss (read back to the host). The full loop — with running-mean train metrics — lives
in [`../resnet_tinyimagenet.jl`](../resnet_tinyimagenet.jl). Swap `using Reactant` for `using CUDA`
there to run the eager CUDA.jl path instead (same steady-state speed, without the one-time compile).

## Notes on faithfulness

- **Augmentation** is vectorized over the batch (one crop offset + flip decision per batch) rather
  than per-image as in the Julia `augment`. This is cheaper and, at batch size 128, statistically
  negligible for this comparison; a per-image variant is straightforward if exactness is wanted.
- **Data layout**: HuggingFace `datasets` with `.with_format("torch")` hands an Image column back
  as an NCHW uint8 tensor already, so no permute is needed (the Julia "julia" format instead yields
  CWHN and permutes to WHCN).
- Normalization and augmentation run on the GPU, matching where the Flux version does the same work.
