# ResNet-18 on Tiny-ImageNet-200 (Flux + HuggingFaceDatasets.jl)

Trains a small-image **ResNet-18** on **Tiny-ImageNet-200** — a 200-class, 64×64 subset of
ImageNet — pulled from the HuggingFace Hub with
[HuggingFaceDatasets.jl](https://github.com/CarloLucibello/HuggingFaceDatasets.jl) and trained on
the GPU with `Flux.train!`.

| file | what |
| --- | --- |
| [`resnet_tinyimagenet.jl`](resnet_tinyimagenet.jl) | the example (data pipeline, model, training loop) |
| [`Project.toml`](Project.toml) | its self-contained environment |
| [`pytorch/`](pytorch/) | a line-for-line PyTorch port + a Flux-vs-PyTorch performance comparison |

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
$ julia --project=. -t auto resnet_tinyimagenet.jl                # 30 epochs (the default)
$ julia --project=. -t auto resnet_tinyimagenet.jl --epochs 5     # override the epoch count
$ julia --project=. -t auto resnet_tinyimagenet.jl --help         # list all options
```

On the **first run**, HuggingFaceDatasets.jl transparently builds a small Python environment
(via CondaPkg: `datasets`, `pillow`, `numpy`) and downloads the dataset — a few minutes, once.
`-t auto` enables threaded batch collation; `--num-workers > 0` (default `4`) spreads the CPython
image decode across worker processes, past the GIL.

Command-line options (parsed with [ArgParse.jl](https://github.com/carlobaldassi/ArgParse.jl) —
run with `--help` for the full list): `--epochs`, `--batchsize`, `--lr`, `--num-workers`. The same
knobs are the keywords of `main`, so it can also be driven from the REPL:
`main(; epochs=30, batchsize=128, lr=1e-3, num_workers=4)`. Each epoch logs train/validation loss
and top-1 accuracy. A small-image ResNet-18 trained this way typically reaches **~50% top-1
validation accuracy** in ~30 epochs (indicative — from-scratch, no pretraining).

