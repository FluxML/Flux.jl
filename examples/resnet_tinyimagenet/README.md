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

- **`caching_allocator=false`.** By default `train!` wraps each step in a cross-step allocation
  cache (`GPUArrays.@cached`) that reuses buffers to cut GC churn. The catch: it keeps *every*
  allocation of a step alive, so the step's peak memory becomes the **sum** of its allocations
  (~2× the true working set here — measured: 16 GiB vs 9 GiB at batch 128). Worse, the **cold first
  step** runs cuDNN's convolution-algorithm search, whose per-algorithm workspaces are retained too,
  ballooning past a 32 GB GPU and OOMing (even at batch 64). Passing `caching_allocator=false`
  restores in-step recycling and keeps the peak at the working set. This option is Flux
  [#2695](https://github.com/FluxML/Flux.jl/pull/2695); until it's in a registered release, the
  `Project.toml` points at this repo's Flux via `[sources]`.
- **GPU memory.** Batch size 128 is comfortable on a 32 GB card. Lower `batchsize` if you have less.
