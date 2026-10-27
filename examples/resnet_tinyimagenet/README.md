# ResNet-18 on Tiny-ImageNet-200

Trains a small-image **ResNet-18** on **Tiny-ImageNet-200** — a 200-class, 64×64 subset of
ImageNet — pulled from the HuggingFace Hub with
[HuggingFaceDatasets.jl](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl). The same training
script comes in three versions, side by side:

| folder | framework | the training step |
| --- | --- | --- |
| [`flux-cuda/`](flux-cuda/) | Flux + [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) | runs eagerly on `CuArray`s, differentiated by Zygote, with cuDNN convolutions |
| [`flux-reactant/`](flux-reactant/) | Flux + [Reactant](https://github.com/EnzymeAD/Reactant.jl) | compiled into a single XLA program by `Flux.trainstep!`, differentiated by Enzyme |
| [`pytorch/`](pytorch/) | PyTorch | a port, for comparison |

The two Flux versions share the data pipeline, the model and the training loop in
[`training.jl`](training.jl); their `train.jl` only choose the device. Each folder has its own
environment: CUDA.jl and Reactant are not installed together.

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
  predictions computed in the forward pass. Each epoch logs the training loss, accuracy, time and
  throughput, and the validation loss and top-1 accuracy.

## Running

```console
$ cd examples/resnet_tinyimagenet/flux-cuda      # or flux-reactant
$ julia --project=. -e 'using Pkg; Pkg.instantiate()'
$ julia --project=. train.jl                     # 30 epochs (the default)
$ julia --project=. train.jl --epochs 5
$ julia --project=. train.jl --help             # list all options

$ cd examples/resnet_tinyimagenet/pytorch
$ uv sync
$ uv run python resnet_tinyimagenet.py
```

The three scripts take the same options: `--epochs`, `--batchsize`, `--lr`, `--weight-decay`,
`--num-workers`, `--seed` and `--clip-norm`, plus `--compile` (`torch.compile`) for PyTorch. They
print the same log lines:

```
[train] epoch=2 lr=7.50e-04 loss=3.4088 acc=0.2233 time=36.6s img/s=2734
[eval] epoch=2 val_loss=3.2746 val_acc=0.2450
```

The first run downloads the dataset (150 MB) into the HuggingFace cache, and the first run of each
Flux version installs its Python environment (with `datasets`) in `.CondaPkg/`. With Reactant,
that environment also holds jax and TensorFlow (2.9 GB), which Reactant's PythonCall extension
imports when it loads: TensorFlow's warnings that it cannot load the CUDA libraries are harmless.
Reactant also caps Python at 3.12.0, whose `multiprocess` prints an ignored `AttributeError` about
`_recursion_count` at exit.

## Performance

Measured on an **RTX 5090 (32 GB)**, driver 595.84, batch size 128, 4 data-loading workers,
3-epoch runs (`--epochs 3`). Julia 1.13.2, Flux 0.16.14, NNlib 0.9.55, Zygote 0.7.15, CUDA.jl
6.4.2 with cuDNN 9.27, Reactant 0.2.300; PyTorch 2.14.1 (cu130) with cuDNN 9.24.

| | s/epoch | images/s | first epoch | peak GPU memory in use | peak GPU memory reserved |
| --- | --- | --- | --- | --- | --- |
| [Flux + CUDA.jl](flux-cuda/) | 36.7 | 2720 | 77.6 s | 8.4 GiB | 8.7 GiB |
| [Flux + Reactant](flux-reactant/) | 30.2 | 3310 | 103.0 s | 5.6 GiB | 23.5 GiB, or 9.2 GiB without preallocation\* |
| [PyTorch](pytorch/) | 28.9 | 3460 | 29.4 s | 4.0 GiB | 4.1 GiB |
| [PyTorch](pytorch/), `--compile` | 25.2 | 3960 | 42.1 s | 3.9 GiB | 4.1 GiB |

*s/epoch* is the steady-state training time of an epoch (99,968 images, without the evaluation),
and *first epoch* includes the compilation of the training step. The GPU memory is the most memory
allocated at once (*in use*) and the most memory held by the allocator (*reserved*) over a 1-epoch
run, as reported by the allocators: CUDA's memory pool, XLA's, and PyTorch's
(`torch.cuda.max_memory_allocated` and `max_memory_reserved`). `nvidia-smi` shows the reserved
memory plus ~0.5 GiB of CUDA context.

\* When Reactant initializes, XLA reserves 75% of the GPU memory, whatever the program needs.
With `XLA_REACTANT_GPU_PREALLOCATE=false` it allocates on demand, at the same speed;
`XLA_REACTANT_GPU_MEM_FRACTION` changes the fraction. See Reactant's
[GPU configuration](https://enzymead.github.io/Reactant.jl/dev/api/config#GPU-Configuration).

- **All four are limited by the GPU**, which is busy 94–97% of the time during training. Float32
  convolutions run on TF32 tensor cores by default in all of them.
- **Reactant** compiles the forward pass, the backward pass and the optimizer update into one XLA
  program. In the steady state it trains 1.2× faster than CUDA.jl and 4% slower than eager
  PyTorch. XLA plans the buffers of the whole step ahead of time: on a synthetic batch, a step
  peaks at 2.6 GiB, as in PyTorch. Over a run, the batches moved to the GPU add up to 3 GiB more,
  because they are freed only when Julia's garbage collector finalizes them: the device iterator
  of MLDataDevices frees the previous batch on CUDA, but not on Reactant. Compiling the training
  step takes ~73 s (the first epoch minus a steady one), and the two evaluation functions (for the
  full and the last validation batch) take ~40 s more before the first evaluation.
- **CUDA.jl** runs every operation of the forward and backward passes eagerly, with cuDNN
  convolutions and batch normalization. Zygote keeps all the intermediate arrays of the forward
  pass for the backward pass, and CUDA.jl frees GPU memory only when Julia's garbage collector
  finalizes the arrays: a step peaks at 8.4 GiB, 3× PyTorch's 2.6 GiB, since PyTorch frees each
  tensor as soon as it is no longer needed. The script runs an incremental garbage collection
  after each training step and each evaluation batch, and a full one after the first step; with
  the collections after the training steps only, the process held 21.6 GiB, at the same speed.
  The first epoch includes ~40 s of Julia compilation.
- **PyTorch** reaches its steady speed in the first epoch; `torch.backends.cudnn.benchmark` times
  the convolution algorithms on the first batches. `--compile` (`torch.compile`) fuses the
  elementwise operations around the convolutions (batch normalization, ReLU, residual additions),
  which makes it 1.15× faster than eager PyTorch, for ~17 s of compilation.

Loading the packages, starting the data-loading workers and the first evaluation take ~1.5 min
before the first epoch with CUDA.jl, ~2 min with Reactant (with the compilation of the evaluation)
and ~25 s with PyTorch. After 3 epochs, the validation accuracy is 34–35% with Flux and 30–31%
with PyTorch (single runs).
