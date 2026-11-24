# ResNet-18 on Tiny-ImageNet-200

Trains a small-image **ResNet-18** on **Tiny-ImageNet-200** — a 200-class, 64×64 subset of
ImageNet — pulled from the HuggingFace Hub with
[HuggingFaceDatasets.jl](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl). The same training
script comes in four versions, side by side:

| folder | framework | the training step |
| --- | --- | --- |
| [`flux-cuda/`](flux-cuda/) | Flux + [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) | runs eagerly on `CuArray`s, differentiated by Zygote, with cuDNN convolutions |
| [`flux-reactant/`](flux-reactant/) | Flux + [Reactant](https://github.com/EnzymeAD/Reactant.jl) | compiled into a single XLA program by `Flux.trainstep!`, differentiated by Enzyme |
| [`lux-reactant-nitro/`](lux-reactant-nitro/) | Lux + [ReactantNitro](https://github.com/EnzymeAD/ReactantNitro.jl) | an experiment, a struct plus hooks, whose training step, evaluation and data transfers Nitro compiles with Reactant; differentiated by Enzyme |
| [`pytorch/`](pytorch/) | PyTorch | a port, for comparison |

The two Flux versions share the model and the training loop in [`training.jl`](training.jl); their
`train.jl` only choose the device. All the Julia versions share the data pipeline in
[`data.jl`](data.jl), and the Lux version writes the model and the training as a Nitro experiment in
its own `train.jl`. Each folder has its own environment: CUDA.jl and Reactant are not installed
together.

The Flux environments use the Flux checkout containing this example, and NNlib 0.9.60 with
Reactant and 0.9.59 with CUDA.jl; the Lux one uses registered releases only (Lux 1.31,
ReactantNitro 0.2, Reactant 0.2.308). The CUDA version takes CUDA.jl and cuDNN.jl from CUDA.jl's
GitHub `main` branch, as autotuning convolutions is not in a cuDNN.jl release yet; both versions
require MLDataDevices 1.18, whose device iterator frees the previous batch on Reactant as well as
on CUDA.

## What it does

- **Data** — [`zh-plus/tiny-imagenet`](https://huggingface.co/datasets/zh-plus/tiny-imagenet)
  (100k train / 10k validation, 64×64 RGB, 200 classes), decoded on the fly by 4 worker processes.
  Every image is forced to 3-channel RGB (a few are grayscale), standardized with the ImageNet
  per-channel mean/std, and — for training — augmented with a random 4-px-pad crop and horizontal
  flip. The last, smaller training batch of each epoch is dropped, so that every training step has
  the same shape.
- **Model** — ResNet-18 adapted for small images: the aggressive 224×224 stem (7×7 stride-2 conv +
  max-pool) is replaced by a single 3×3 stride-1 conv so a 64×64 image isn't over-downsampled. The
  rest is stock ResNet-18 — four stages of two BasicBlocks (64→128→256→512), global average pool,
  linear head (11.27M params).
- **Training** — `AdamW` with a cosine-annealed learning rate, cross-entropy on the integer class
  labels, and an epoch loop over `Flux.trainstep!`, which also returns the number of correct
  predictions computed in the forward pass and manages training garbage collection. The CUDA
  version autotunes its cuDNN convolutions, and also runs an incremental collection after each
  evaluation batch. Each epoch logs the
  training loss, accuracy, time and throughput, and the validation loss and top-1 accuracy.

## Running

```console
$ cd examples/resnet_tinyimagenet/flux-cuda      # or flux-reactant
$ julia --project=. -e 'using Pkg; Pkg.instantiate()'
$ julia --project=. train.jl                     # 30 epochs (the default)
$ julia --project=. train.jl --epochs 5
$ julia --project=. train.jl --help             # list all options

$ cd examples/resnet_tinyimagenet/lux-reactant-nitro
$ julia --project=. -e 'using Pkg; Pkg.instantiate()'
$ julia --threads=1,0 --project=. train.jl --epochs 5

$ cd examples/resnet_tinyimagenet/pytorch
$ uv sync
$ uv run python resnet_tinyimagenet.py
```

The four scripts take the same options: `--epochs`, `--batchsize`, `--lr`, `--weight-decay`,
`--num-workers`, `--seed` and `--clip-norm`, plus `--compile` (`torch.compile`) for PyTorch. They
print the same log lines:

```
[train] epoch=2 lr=7.50e-04 loss=3.4250 acc=0.2194 time=30.2s img/s=3315
[eval] epoch=2 val_loss=3.2333 val_acc=0.2490
```

The first run downloads the dataset (150 MB) into the HuggingFace cache, and the first run of each
Julia version installs its Python environment (with `datasets`) in `.CondaPkg/`. With Reactant,
that environment also holds jax and TensorFlow (2.9 GB), which Reactant's PythonCall extension
imports when it loads: TensorFlow's warnings that it cannot load the CUDA libraries are harmless.
Reactant also caps Python at 3.12.0, whose `multiprocess` prints an ignored `AttributeError` about
`_recursion_count` at exit.

## Performance

Measured on 12 November 2026 on an **RTX 5090 (32 GB)**, driver 595.99, batch size 128,
4 data-loading workers, 3-epoch runs (`--epochs 3`), except for eager PyTorch, measured on
6 November 2026 with driver 595.84. Julia 1.13.2 (4 threads) and MLDataDevices 1.18.0. Both Flux
versions used Flux master
([`520788b`](https://github.com/FluxML/Flux.jl/commit/520788b44946473fcf27ca32f7899bd757591527));
Flux + CUDA.jl used NNlib 0.9.59, Zygote 0.7.18, and CUDA.jl and cuDNN.jl main
([`3d6a352`](https://github.com/JuliaGPU/CUDA.jl/commit/3d6a3523d01d0e8c227e5cea499d27a3ea04c342))
with cuDNN 9.28, and Flux + Reactant used NNlib 0.9.60 and Reactant 0.2.308. PyTorch 2.14.1
(cu130) with cuDNN 9.24. The Lux version ran on Julia 1.13.2 with `--threads=1,0`, Lux 1.31.6,
ReactantNitro 0.2.2, Optimisers 0.4.11 and Reactant 0.2.308.

| | s/epoch | images/s | first epoch |
| --- | --- | --- | --- |
| [Flux + CUDA.jl](flux-cuda/) | 30.1 | 3320 | 63.0 s |
| [Flux + Reactant](flux-reactant/) | 25.2 | 3970 | 88.0 s |
| [Lux + ReactantNitro](lux-reactant-nitro/) | 25.3 | 3960 | 52.5 s |
| [PyTorch](pytorch/) | 28.9 | 3450 | 29.4 s |
| [PyTorch](pytorch/), `--compile` | 25.3 | 3960 | 42.0 s |

*s/epoch* averages epochs 2 and 3 (99,968 images per epoch, without evaluation), and *first epoch*
includes compilation of the training step. The `torch.compile` run starts with empty Inductor and
Triton caches.

Memory was measured in the same runs, including the initial evaluation, training and final
evaluation.

| | peak GPU memory in use | peak GPU memory reserved |
| --- | --- | --- |
| [Flux + CUDA.jl](flux-cuda/) | 8.4 GiB | 9.4 GiB |
| [Flux + Reactant](flux-reactant/) | 3.0 GiB | 8.0 GiB\* |
| [Lux + ReactantNitro](lux-reactant-nitro/) | 2.8 GiB | 6.0 GiB\* |
| [PyTorch](pytorch/) | 4.0 GiB | 4.1 GiB |
| [PyTorch](pytorch/), `--compile` | 3.9 GiB | 4.1 GiB |

These are the most memory allocated at once (*in use*) and held by the allocator (*reserved*):
CUDA's pool high-water marks,
XLA's peak bytes in use and peak pool bytes, and PyTorch's `max_memory_allocated` and
`max_memory_reserved`. `nvidia-smi` also includes the CUDA context.

\* The Reactant runs, timings included, set `XLA_REACTANT_GPU_PREALLOCATE=false`, allocating on
demand; with the default preallocation, Flux + Reactant trains at the same speed. By default, XLA
reserves 75% of GPU memory; `XLA_REACTANT_GPU_MEM_FRACTION` changes the fraction.
See Reactant's
[GPU configuration](https://enzymead.github.io/Reactant.jl/dev/api/config#GPU-Configuration).

- **All five use Float32 models**, with TF32 tensor cores enabled for cuDNN convolutions.
- **Reactant** compiles the forward pass, the backward pass and the optimizer update into one XLA
  program. It trains 1.19× faster than CUDA.jl and 1.15× faster than eager PyTorch. NNlib 0.9.60
  computes BatchNorm's batch statistics in one pass under Reactant, and drops a zero gradient
  term through the mean from the backward pass, as Lux does
  ([NNlib.jl#920](https://github.com/FluxML/NNlib.jl/pull/920)): with NNlib 0.9.59, an epoch
  took 26.4 s, and 30.6 s on 28 October with Reactant 0.2.301. The Lux version sped up as much
  between those Reactant versions, which points to Reactant, but the cause has not been
  investigated. MLDataDevices 1.18 frees the previous batch on Reactant, and the run peaks at
  3.0 GiB in use. Compilation adds about 63 s to the first training epoch; evaluation also
  compiles separate functions for full and partial batches.
- **Lux + ReactantNitro** compiles the same pieces with Reactant, through Nitro, which also compiles
  the evaluation, padding the last, smaller validation batch to the shape of the others and slicing
  the outputs back, and moves each batch to the device and frees it. It trains as fast as Flux +
  Reactant and compiled PyTorch (1.13× faster than on 28 October, 28.5 s per epoch). Compilation
  adds about 27 s to the first training epoch, and the run peaks at 2.8 GiB in use. The model, data
  and optimizer are the same, with these differences: Nitro's weight decay is decoupled but does not
  decay biases and normalization parameters (irrelevant at the default `--weight-decay 0`),
  `--clip-norm` clips the global gradient norm to 10 instead of each parameter array separately, and
  the script runs with `--threads=1,0`, since PythonCall, which reads the dataset, segfaults on the
  threads that Nitro uses for its data tasks otherwise.
- **CUDA.jl** runs every operation of the forward and backward passes eagerly, with cuDNN
  convolutions and batch normalization. With `cuDNN.autotune!(true)`, the first call of each
  convolution shape times the plans that cuDNN supports and keeps the fastest, as
  `torch.backends.cudnn.benchmark` does, instead of the first plan proposed by cuDNN's
  heuristics. This makes training 1.14× faster (34.2 s per epoch without it, in the same
  environment), and raises the peak memory reserved from 8.7 to 9.4 GiB, while the peak in use
  stays at 8.4 GiB: CUDA.jl's memory pool keeps the workspaces that autotuning allocates (up to
  1.6 GiB, for the largest candidate of a shape), and the fastest plans need larger workspaces.
  Flux 0.16.15 manages garbage collection inside
  `trainstep!`, including a full collection after a step that compiles; the script only adds
  incremental collections during evaluation. The first training epoch takes about 33 s longer
  than a steady one, including Julia compilation and autotuning.
- **PyTorch** reaches its steady speed in the first epoch; `torch.backends.cudnn.benchmark` times
  the convolution algorithms on the first batches. `--compile` (`torch.compile`) fuses the
  elementwise operations around the convolutions (batch normalization, ReLU, residual additions),
  which makes it 1.14× faster than eager PyTorch, for ~17 s of compilation.

After 3 epochs, validation accuracy is 34.1% with CUDA.jl, 34.3% with Reactant (33.2% and 34.3% in
two runs with NNlib 0.9.59), 34.1% with Lux and ReactantNitro, 31.0% with eager PyTorch and 30.6%
with compiled PyTorch (single runs otherwise). The defaults initialize weights differently:
`glorot_uniform` in Flux and Lux, and in PyTorch a `kaiming_uniform` that makes the weights of most
3×3 convolutions ~1.7× smaller. See the [PyTorch port's
notes](pytorch/README.md#notes-on-faithfulness) for the other differences between the data
pipelines.
