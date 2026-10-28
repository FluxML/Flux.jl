# GPT-2 pretraining with Flux and Reactant

The reference version of the [GPT-2 example](../README.md), which describes the model, data,
training and presets. Here the training step is compiled to a single XLA program with
[Reactant](https://github.com/EnzymeAD/Reactant.jl) through `Flux.trainstep!`, so no CUDA.jl is
needed. The data comes from the HuggingFace Hub through
[HuggingFaceDatasets.jl](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl) and is tokenized with
[HuggingFaceTokenizers.jl](https://github.com/MurrellGroup/HuggingFaceTokenizers.jl).

| file | what |
| --- | --- |
| [`data.jl`](data.jl) | streams a dataset from the Hub and tokenizes it into a cached dataset of token blocks; the batch loader |
| [`model.jl`](model.jl) | the GPT-2 model, the loss, and sampling |
| [`hellaswag.jl`](hellaswag.jl) | [HellaSwag](https://rowanzellers.com/hellaswag) accuracy, during training or for a saved checkpoint |
| [`training.jl`](training.jl) | the training loop: presets, gradient accumulation, evaluation, checkpoints, command-line options |
| [`train.jl`](train.jl) | the training script: runs the loop on Reactant |
| [`Project.toml`](Project.toml) | the environment |

All the files but `train.jl` are independent of the device, and [`../flux-cuda`](../flux-cuda/)
includes them.

## Running

```console
$ cd examples/gpt2_fineweb/flux-reactant
$ julia --project=. -e 'using Pkg; Pkg.instantiate()'

$ julia --project=. train.jl --preset tinystories   # ~30 min on an RTX 5090
$ julia --project=. train.jl --preset fineweb
$ julia --project=. train.jl --preset fineweb --resume log/checkpoint_05000.jld2
$ julia --project=. hellaswag.jl log/checkpoint_19073.jld2
```

The first run with a preset streams and tokenizes its dataset into `../data/<dataset>/`: ~1 min
and 0.95 GB for TinyStories, ~30 min (28.5 GB streamed) and 20 GB for FineWeb-Edu.

Options (`--help` lists them all): `--batchsize` is the number of sequences per micro-batch (the
preset's tokens per step are reached by accumulating gradients, so it only trades speed for
memory), `--steps`, `--lr`, `--eval-every`, `--hellaswag-every`, `--ckpt-every` (0 disables
each), `--precision` (see below), `--resume`, `--seed`, `--data-dir`, `--log-dir`, and
`--peak-tflops` to report the model FLOPs utilization. Every line printed is also appended to
`log/log.txt`.

## Results

The [shared comparison](../README.md#performance) records current GPT-2 124M timings and memory
measurements on the RTX 5090. NNlib master fixes the LayerNorm activation promotion that slowed
down bfloat16 training with Flux 0.16.14 and NNlib 0.9.55.

The full TinyStories run below used Julia 1.13, Flux 0.16.13, NNlib 0.9.50 and Reactant 0.2.295;
it has not been repeated with the current environment.

**`tinystories`** — 22 minutes of training (0.18 s per step, ~365k tokens/s) after ~3 minutes of
compilation. The validation loss goes 3.21 → 2.01 → 1.67 → **1.488** at steps 250, 1000, 3000 and
7232 (1.486 with `Float32` parameters, which took 29 minutes), and the samples are fluent:

> Hello, I'm a language model, Ben. You can talk or sing or laugh. That is nice. Can you teach
> me?" Anna said, "Yes, I can. I can…

The HellaSwag evaluation reproduces build-nanogpt's numbers for OpenAI's GPT-2 (124M) weights
loaded into this model: acc = 0.2861, acc_norm = 0.2956 (reference: 0.2859, 0.2955).

## Notes

- **Precision.** With `--precision bf16` (the default) the model is converted with `Flux.bf16`:
  weights and activations are bfloat16, LayerNorm parameters stay in `Float32`, the optimizer
  keeps and updates `Float32` master weights (`MixedPrecision(Float32, rule)`), and the loss is
  computed in `Float32`. NNlib master keeps LayerNorm's output in bfloat16 too, fixing the
  activation promotion with NNlib 0.9.55. `tf32` keeps everything in `Float32` with XLA's default
  matmul precision on NVIDIA GPUs (TF32-like), and `f32` is exact
  (`Reactant.with_config(; dot_general_precision = PrecisionConfig.HIGHEST)` around the training
  loop, as the setting is read when a step is compiled).
- **Attention.** Under Reactant, attention goes through NNlib's standard implementation, which
  materializes the `(T, T, heads, batch)` attention matrix: there is no fused (flash) attention
  for traced arrays yet ([Reactant.jl#3510](https://github.com/EnzymeAD/Reactant.jl/issues/3510)).
  This bounds the micro-batch size and costs speed; the example uses `Attention(...; is_causal=true)`,
  so it will benefit from a fused implementation without changes.
- **Compilation.** The first training step includes tracing, XLA compilation and Julia compiling
  the function that calls the compiled step. Its compile time grows quadratically with the number
  of parameter and optimizer-state arrays
  ([Reactant.jl#3511](https://github.com/EnzymeAD/Reactant.jl/issues/3511)). The validation loss
  and HellaSwag scoring are compiled once each too, by `Reactant.compile`, when enabled.
- **Memory.** XLA preallocates a fraction of the GPU memory; the script raises it from 0.75 to 0.9
  (`XLA_REACTANT_GPU_MEM_FRACTION`). Set `XLA_REACTANT_GPU_PREALLOCATE=false` to let the pool grow
  as needed, as in the shared benchmark.
- **Gradient clipping** uses `ClipNorm(1.0)`, which clips each parameter array separately rather than
  the global norm as build-nanogpt does. The PyTorch port clips per parameter too.
- **Python.** HuggingFaceDatasets.jl and HuggingFaceTokenizers.jl run the Python `datasets` and
  `tokenizers` packages, installed automatically by CondaPkg in `.CondaPkg/`. The scripts load
  PythonCall before Flux, as on Julia 1.13 the Python environment otherwise resolves differently
  ([JuliaLang/julia#63967](https://github.com/JuliaLang/julia/issues/63967)).
