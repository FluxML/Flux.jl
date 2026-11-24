# Models and Examples

## [Pretrained models with Metalhead.jl](@id man-pretrained-models)

Metalhead.jl provides pretrained Flux models for vision, language and image generation. Install Metalhead 0.10 or later with `] add Metalhead`, then load an ImageNet-pretrained ResNet.

```julia
using Metalhead

model = create_model("timm/resnet50.a1_in1k")
```

Browse the [Metalhead model catalog](https://fluxml.ai/Metalhead.jl/stable/models/) for architectures and checkpoints. For image preprocessing, inference and transfer learning, follow the [Metalhead vision tutorial](https://fluxml.ai/Metalhead.jl/stable/tutorials/vision/).

## End-to-end training

- [ResNet-18 on TinyImageNet](https://github.com/FluxML/Flux.jl/tree/master/examples/resnet_tinyimagenet) trains an image classifier and validates it on held-out images.
- [GPT-2 pretraining](https://github.com/FluxML/Flux.jl/tree/master/examples/gpt2_fineweb) includes gradient accumulation, mixed precision and checkpointing.

Both examples include a Flux implementation using [Reactant](../guide/reactant.md) to compile the training step.

## Model Zoo

The [model zoo](https://github.com/FluxML/model-zoo) is a collection of examples that demonstrate how to build and train models using Flux. The examples are organised by domain and include vision, text, and audio. Each example includes a description of the model, the data used, and the training process.

Some of the examples are pedagogical, see for instance
- [Multilayer Perceptron](https://github.com/FluxML/model-zoo/tree/master/vision/mlp_mnist)
- [Simple Convolutional Neural Network](https://github.com/FluxML/model-zoo/tree/master/vision/conv_mnist)

Others are more advanced, see for instance
- [Variational Autoencoder](https://github.com/FluxML/model-zoo/blob/master/vision/vae_mnist)
