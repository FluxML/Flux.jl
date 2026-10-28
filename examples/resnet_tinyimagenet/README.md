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

The Flux environments use the Flux checkout containing this example and NNlib's GitHub `master`
branch; the Lux one uses registered releases only (Lux 1.31, ReactantNitro 0.2, Reactant 0.2.301). The CUDA version requires CUDA.jl and cuDNN.jl 6.5; both versions require MLDataDevices
1.18, whose device iterator frees the previous batch on Reactant as well as on CUDA.

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
  version also runs an incremental collection after each evaluation batch. Each epoch logs the
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
[train] epoch=2 lr=7.50e-04 loss=3.4377 acc=0.2192 time=35.2s img/s=2840
[eval] epoch=2 val_loss=3.2136 val_acc=0.2544
```

The first run downloads the dataset (150 MB) into the HuggingFace cache, and the first run of each
Julia version installs its Python environment (with `datasets`) in `.CondaPkg/`. With Reactant,
that environment also holds jax and TensorFlow (2.9 GB), which Reactant's PythonCall extension
imports when it loads: TensorFlow's warnings that it cannot load the CUDA libraries are harmless.
Reactant also caps Python at 3.12.0, whose `multiprocess` prints an ignored `AttributeError` about
`_recursion_count` at exit.

## Performance

Measured on 28 October 2026 on an **RTX 5090 (32 GB)**, driver 595.84, batch size 128,
4 data-loading workers, 3-epoch runs (`--epochs 3`). Julia 1.13.2 (4 threads for timings,
32 for memory profiles),
Flux 0.16.15, NNlib master
([`b184469`](https://github.com/FluxML/NNlib.jl/commit/b18446979d1b7b8dc03288f64f441b1c3b55d9b2)),
MLDataDevices 1.18.0, Zygote 0.7.15, CUDA.jl and cuDNN.jl 6.5.0 with cuDNN 9.27,
Reactant 0.2.301; PyTorch 2.14.1 (cu130) with cuDNN 9.24. The Lux version ran on Julia 1.13.2
with `--threads=1,0`, Lux 1.31.5, ReactantNitro 0.2.1, Optimisers 0.4.11 and Reactant 0.2.301.

| | s/epoch | images/s | first epoch |
| --- | --- | --- | --- |
| [Flux + CUDA.jl](flux-cuda/) | 35.3 | 2830 | 67.3 s |
| [Flux + Reactant](flux-reactant/) | 30.6 | 3270 | 101.5 s |
| [Lux + ReactantNitro](lux-reactant-nitro/) | 28.5 | 3510 | 58.1 s |
| [PyTorch](pytorch/) | 28.9 | 3450 | 29.4 s |
| [PyTorch](pytorch/), `--compile` | 25.3 | 3950 | 42.6 s |

*s/epoch* averages epochs 2 and 3 (99,968 images per epoch, without evaluation), and *first epoch*
includes compilation of the training step. The `torch.compile` run starts with empty Inductor and
Triton caches.

Memory was measured over the following run lengths, including the initial evaluation, training
and final evaluation. The one-epoch Flux profiles were separate runs from the timing runs above.

| | epochs measured | peak GPU memory in use | peak GPU memory reserved |
| --- | --- | --- | --- |
| [Flux + CUDA.jl](flux-cuda/) | 1 | 8.4 GiB | 8.7 GiB |
| [Flux + Reactant](flux-reactant/) | 1 | 2.6 GiB | 5.2 GiB\* |
| [Flux + Reactant](flux-reactant/) | 3 | 4.4 GiB | not recorded |
| [Lux + ReactantNitro](lux-reactant-nitro/) | 1 | 2.9 GiB | 5.5 GiB\* |
| [PyTorch](pytorch/) | 3 | 4.0 GiB | 4.1 GiB |
| [PyTorch](pytorch/), `--compile` | 3 | 3.9 GiB | 4.1 GiB |

These are the most memory allocated at once (*in use*) and held by the allocator (*reserved*):
CUDA's pool high-water marks,
XLA's peak bytes in use and peak pool bytes, and PyTorch's `max_memory_allocated` and
`max_memory_reserved`. `nvidia-smi` also includes the CUDA context.

\* Reactant is measured with `XLA_REACTANT_GPU_PREALLOCATE=false`, allocating on demand.
By default, XLA reserves 75% of GPU memory; `XLA_REACTANT_GPU_MEM_FRACTION` changes the fraction.
See Reactant's
[GPU configuration](https://enzymead.github.io/Reactant.jl/dev/api/config#GPU-Configuration).

- **All five use Float32 models**, with TF32 tensor cores enabled for cuDNN convolutions.
- **Reactant** compiles the forward pass, the backward pass and the optimizer update into one XLA
  program. It trains 1.15× faster than CUDA.jl and 6% slower than eager PyTorch in these runs.
  MLDataDevices 1.18 frees the previous batch on Reactant. The one-epoch profile peaks at
  2.6 GiB in use, compared with 5.6 GiB in the previous measurement with MLDataDevices 1.17.10.
  The three-epoch run peaks at 4.4 GiB; the difference between these runs has not been investigated,
  so the one-epoch result does not establish stable memory use across epochs.
  Compilation adds about 71 s to the first training epoch; evaluation also compiles
  separate functions for full and partial batches.
- **Lux + ReactantNitro** compiles the same pieces with Reactant, through Nitro, which also
  compiles the evaluation, padding the last, smaller validation batch to the shape of the others
  and slicing the outputs back, and moves each batch to the device and frees it. It trains 7%
  faster than Flux + Reactant and 1% faster than eager PyTorch; why it is faster than Flux has not
  been investigated. Compilation adds about 30 s to the first training epoch, and the one-epoch
  profile peaks at 2.9 GiB in use. The model, data and optimizer are the same, with these
  differences: Nitro's weight decay is decoupled but does not decay biases and normalization
  parameters (irrelevant at the default `--weight-decay 0`), `--clip-norm` clips the global
  gradient norm to 10 instead of each parameter array separately, and the script runs with
  `--threads=1,0`, since PythonCall, which reads the dataset, segfaults on the threads that Nitro
  uses for its data tasks otherwise.
- **CUDA.jl** runs every operation of the forward and backward passes eagerly, with cuDNN
  convolutions and batch normalization. Flux 0.16.15 manages garbage collection inside
  `trainstep!`, including a full collection after a step that compiles; the script only adds
  incremental collections during evaluation. The first training epoch takes about 32 s longer
  than a steady one, including Julia compilation and cuDNN's algorithm search.
- **PyTorch** reaches its steady speed in the first epoch; `torch.backends.cudnn.benchmark` times
  the convolution algorithms on the first batches. `--compile` (`torch.compile`) fuses the
  elementwise operations around the convolutions (batch normalization, ReLU, residual additions),
  which makes it 1.14× faster than eager PyTorch, for ~17 s of compilation.

After 3 epochs, validation accuracy is 34.3% with CUDA.jl, 34.6% with Reactant, 34.8% with Lux and
ReactantNitro, 30.4% with eager PyTorch and 31.4% with compiled PyTorch (single runs). The defaults
initialize weights differently: `glorot_uniform` in Flux and Lux, and in PyTorch a `kaiming_uniform` that makes the weights of most 3×3
convolutions ~1.7× smaller. See the [PyTorch port's notes](pytorch/README.md#notes-on-faithfulness)
for the other differences between the data pipelines.
