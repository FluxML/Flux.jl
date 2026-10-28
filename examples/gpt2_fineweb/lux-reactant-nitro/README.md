# GPT-2 pretraining with Lux and ReactantNitro

The [GPT-2 example](../README.md) with [Lux](https://github.com/LuxDL/Lux.jl) models and
[ReactantNitro.jl](https://github.com/EnzymeAD/ReactantNitro.jl), a training framework built on
Reactant. The model, the data, the presets, the optimizer and the schedule are those of the
[Flux version](../flux-reactant/). What changes is who runs the training: here the experiment is a
struct plus a handful of hooks, and Nitro compiles the training step (forward, backward, gradient
accumulation and optimizer update), the validation and the data transfers to XLA programs, and owns
the loop, the schedules, the checkpoints and the resume.

| file | what |
| --- | --- |
| [`model.jl`](model.jl) | the GPT-2 model as Lux layers, the loss, and sampling |
| [`experiment.jl`](experiment.jl) | the experiment: presets, data, model, loss, optimizer, schedule and checkpoints |
| [`hellaswag.jl`](hellaswag.jl) | [HellaSwag](https://rowanzellers.com/hellaswag) accuracy, during training or for a saved checkpoint |
| [`train.jl`](train.jl) | the training script: log lines, samples, HellaSwag and the command-line options |
| [`Project.toml`](Project.toml) | the environment |

The data pipeline is the one of [`../flux-reactant/data.jl`](../flux-reactant/data.jl), which this
version includes: it streams, tokenizes and caches the dataset in `../data/`, shared with the other
versions.

## Running

```console
$ cd examples/gpt2_fineweb/lux-reactant-nitro
$ julia --project=. -e 'using Pkg; Pkg.instantiate()'

$ julia --threads=1,0 --project=. train.jl --preset tinystories
$ julia --threads=1,0 --project=. train.jl --preset fineweb
$ julia --threads=1,0 --project=. train.jl --preset fineweb --resume log/<checkpoint>.jld2
$ julia --threads=1,0 --project=. hellaswag.jl log/<checkpoint>.jld2
```

The environment takes the registered releases of Lux, ReactantNitro and Reactant, with no
development version of any package. Run Julia with **`--threads=1,0`**: with an interactive thread,
which is Julia's default, Nitro's data tasks run off the main thread, and PythonCall, which reads
the datasets, segfaults when called from any thread but the main one.

The options are those of the other versions (`--help` lists them), with these differences:

- **Epochs.** Nitro validates, logs, checkpoints and plans its schedules per epoch, so an epoch is
  the `--eval-every` optimizer steps between two validations, and training reads the next window of
  the token stream in each. `--steps` is rounded down to a whole number of epochs (19000 instead of
  19073 for `fineweb`, 7000 instead of 7232 for `tinystories`), and `--hellaswag-every` and
  `--ckpt-every` must be multiples of `--eval-every`. With `--eval-every 0` there is one epoch and no
  validation.
- **Checkpoints.** Nitro writes `epoch-*.jld2` records with the parameters, the optimizer state and
  the position in training, keeping the last and every `--ckpt-every` steps. `--resume` takes one,
  and continues from the next epoch, with the data and the schedules where they were.

## Results

The [shared comparison](../README.md#performance) records the GPT-2 124M timings and memory on the
RTX 5090: 138k tokens/s in steady state, with a first step of 98 s that includes compilation.
The runs below used Julia 1.13.2, Lux 1.31.5, ReactantNitro 0.2.1 and Reactant 0.2.301.

**`tinystories`** — 19 minutes in total with the default options: 7000 steps at 0.1375 s on
average (~477k tokens/s), their compilation, and the 28 validations with samples. The validation
loss goes 3.209 → 2.033 → 1.665 → **1.498** at steps 250, 1000, 3000 and 7000, as in the Flux
version (3.21, 2.01, 1.67 and 1.488 at step 7232, with the schedule of 7232 steps), and the
samples are fluent:

> Hello, I'm a language model, I'm a famous singer!" Tom asked, with a big smile on his face.
> "That's wonderful, Tom!" Anna said, smiling back.

The HellaSwag evaluation reproduces build-nanogpt's numbers for OpenAI's GPT-2 (124M) weights
loaded into this model, without the vocabulary padding: acc = 0.2861, acc_norm = 0.2957
(reference: 0.2859, 0.2955).

## Notes

- **Model.** The layers are Lux's `Dense`, `Embedding` and `LayerNorm`, and the attention is a
  Lux layer over NNlib's `dot_product_attention` with a causal mask, with the queries, keys and
  values in one projection. Like the Flux version under Reactant, it materializes the
  `(T, T, heads, batch)` attention matrix
  ([Reactant.jl#3510](https://github.com/EnzymeAD/Reactant.jl/issues/3510)). The initialization is
  GPT-2's: N(0, 0.02) weights, zero biases, and the projections that feed the residual stream
  scaled by `1/√(2·n_layer)`.
- **Precision.** Nitro has no mixed-precision support, so `--precision bf16` (the default) does it
  in `forward`: the model sees a bfloat16 copy of the parameters, but for the LayerNorm ones, which
  stay in `Float32`, and the optimizer keeps and updates the `Float32` parameters. LayerNorm and
  the loss are computed in `Float32`. `tf32` and `f32` are as in the Flux version.
- **Optimizer.** AdamW is Adam followed by Nitro's decoupled weight decay (`Decay`), which does
  not decay the 1-d parameters, biases and LayerNorm, as in the Flux version. Gradient clipping is
  by the global norm of the accumulated gradient, as in build-nanogpt, unlike the per-array
  `ClipNorm` of the Flux and PyTorch versions.
- **Log lines.** They are the same as in the other versions, printed by a Nitro logger and a phase
  monitor, but the loss of a `[train]` line is that of the last micro-batch of the step, not the
  mean over the step.
