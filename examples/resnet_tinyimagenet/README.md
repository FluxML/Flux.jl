# ResNet-18 on Tiny-ImageNet-200 (Flux + HuggingFaceDatasets.jl)

Trains a small-image **ResNet-18** on **Tiny-ImageNet-200** — a 200-class, 64×64 subset of
ImageNet — pulled from the HuggingFace Hub with
[HuggingFaceDatasets.jl](https://github.com/CarloLucibello/HuggingFaceDatasets.jl) and trained on
the GPU with `Flux.train!`.

| file | what |
| --- | --- |
| [`resnet_tinyimagenet.jl`](resnet_tinyimagenet.jl) | the example (data pipeline, model, training loop) |
| [`Project.toml`](Project.toml) | its self-contained environment |

## What it does

- **Data** — [`zh-plus/tiny-imagenet`](https://huggingface.co/datasets/zh-plus/tiny-imagenet)
  (100k train / 10k validation, 64×64 RGB, 200 classes), decoded on the fly. Every image is forced
  to 3-channel RGB (a few are grayscale), standardized with the ImageNet per-channel mean/std, and —
  for training — augmented with a random 4-px-pad crop and horizontal flip.
- **Model** — ResNet-18 adapted for small images: the aggressive 224×224 stem (7×7 stride-2 conv +
  max-pool) is replaced by a single 3×3 stride-1 conv so a 64×64 image isn't over-downsampled. The
  rest is stock ResNet-18 — four stages of two BasicBlocks (64→128→256→512), global average pool,
  linear head (~11.3M params).
- **Training** — `AdamW`, cross-entropy, `Flux.train!`, on the GPU (falls back to CPU if CUDA is
  unavailable).

## Running

```console
$ cd examples/resnet_tinyimagenet
$ julia --project=. -e 'using Pkg; Pkg.instantiate()'
$ julia --project=. -t auto resnet_tinyimagenet.jl            # 30 epochs (the default)
$ EPOCHS=5 julia --project=. -t auto resnet_tinyimagenet.jl   # override the epoch count
```

On the **first run**, HuggingFaceDatasets.jl transparently builds a small Python environment
(via CondaPkg: `datasets`, `pillow`, `numpy`) and downloads the dataset — a few minutes, once.
`-t auto` enables threaded batch collation; `num_workers > 0` (default `4`) spreads the CPython
image decode across worker processes, past the GIL.

Tunable from the `main` function: `main(; epochs=30, batchsize=128, lr=1e-3, num_workers=4)`.
Each epoch logs train/validation loss and top-1 accuracy. A small-image ResNet-18 trained this way
typically reaches **~50% top-1 validation accuracy** in ~30 epochs (indicative — from-scratch, no
pretraining).

## Notes

- **Memory: handled by the `train!` defaults.** `train!` defaults to `caching_allocator = false`
  with `gc_interval = :auto` — no cross-step buffer cache, plus an adaptive paced GC — which is
  exactly right for a deep conv net, so this example passes no memory keywords. The alternative
  `caching_allocator = true` reuses buffers with a `GPUArrays.AllocCache` (issue
  [#2523](https://github.com/FluxML/Flux.jl/issues/2523),
  [#2695](https://github.com/FluxML/Flux.jl/pull/2695)) but *pins* every allocation of a step, so
  its peak becomes the **sum** of a step's allocations rather than its working set. Measured on an
  RTX 5090 at batch 128, the defaults hold peak at **~8 GiB live / ~9 GiB reserved vs ~23 GiB** with
  the cache, at the **same** time/epoch (the step is compute-bound, so the adaptive GC collects
  every step and is hidden). See [`perf/caching_allocator`](../../perf/caching_allocator) for the
  full comparison. These defaults need this repo's Flux (unreleased), so `Project.toml` points at
  it via `[sources]`.
- **GPU memory.** Batch 128 needs ~9 GiB and is comfortable on a 32 GB card. Lower `batchsize` if
  you have less.
