# PyTorch port — ResNet-18 on Tiny-ImageNet-200

A line-for-line PyTorch translation of the Flux example ([`../training.jl`](../training.jl)), kept
deliberately close so the frameworks can be compared head to head on the same GPU and the same data;
see the [comparison](../README.md#performance).

Same everything but the initialization (see below): small-image ResNet-18 (3×3 stride-1 stem,
four 2-block stages 64→128→256→512, global average pool, linear head, **11.27M params**), the same
`zh-plus/tiny-imagenet` dataset pulled through HuggingFace `datasets`, the same ImageNet
normalization, pad-4 random-crop + horizontal-flip augmentation, `AdamW(lr=1e-3)` with
**`weight_decay=0`** (Flux's `AdamW` default; PyTorch's is `0.01`) and the same cosine
learning-rate schedule, cross-entropy loss, the same command-line options and log lines.

## Running

```console
$ cd examples/resnet_tinyimagenet/pytorch
$ uv sync                                              # torch (cu130) + datasets
$ uv run python resnet_tinyimagenet.py                 # 30 epochs (the default)
$ uv run python resnet_tinyimagenet.py --epochs 3 --compile
$ uv run python resnet_tinyimagenet.py --help
```

`--compile` wraps the model in `torch.compile` (TorchInductor). The `torch` dependency is pinned to
the CUDA 13.0 wheel index (`cu130`), which supports recent NVIDIA GPUs (Blackwell / RTX 50-series
included). Drop the `[tool.uv.sources]` / `[[tool.uv.index]]` blocks from `pyproject.toml` to fall
back to the default wheels.

## Notes on faithfulness

- **Initialization** follows each framework's defaults: Flux's `Conv` and `Dense` use
  `glorot_uniform` weights and zero biases, PyTorch's `Conv2d` and `Linear` use
  `kaiming_uniform(a=√5)`, whose weights in most 3×3 convolutions are ~1.7× smaller. In the
  three-epoch runs, PyTorch reaches 30.4–31.4% validation accuracy and Flux 34.3–34.6%.
  Use `nn.init.xavier_uniform_` weights and zero biases to match Flux's initialization.
- **Augmentation** runs on the GPU and is vectorized over the batch (one crop offset + flip decision
  per batch) rather than per image as in the Julia `augment`, which runs on the CPU in the
  data-loading workers. This is cheaper, and per-image augmentation changes the validation
  accuracy after 3 epochs by less than the run-to-run variation.
- **Data layout**: HuggingFace `datasets` with `.with_format("torch")` hands an Image column back
  as an NCHW uint8 tensor already, so no permute is needed (the Julia "julia" format instead yields
  CWHN and permutes to WHCN).
- **Train metrics** are accumulated on the GPU and read back once per epoch, whereas `Flux.trainstep!`
  returns the loss and the number of correct predictions to the host at every step.
- `torch.backends.cudnn.benchmark = True` lets cuDNN time several convolution algorithms on the first
  batches and pick the fastest.
