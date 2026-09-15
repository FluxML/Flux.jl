# [Migrating from PyTorch](@id man-pytorch)

PyTorch uses different naming conventions and defaults from Flux. This page (while not yet comprehensive) is a guide to navigating
these differences and migrating your PyTorch code to Flux.

## Layers

`torch.Sequential` => `Flux.Chain`

`torch.Linear` => `Flux.Dense`

`torch.Conv1d` => `Flux.CrossCorr((M,)...)`
`torch.Conv2d` => `Flux.CrossCorr((M, N)...)`
`torch.Conv3d` => `Flux.CrossCorr((M, N, O)...)`

PyTorch Conv layers confusingly perform cross-correlation, not convolution. In practice this doesn't often matter. Cross-
correlation is generally used because it may be more efficient as it avoids flipping the kernel. But if you want your network
to be exactly the same as a PyTorch network, use this.

`torch.Upsample` (deprecated) or `torch.nn.functional.interpolate` with `align_corners=True` => `NNlib.Upsample`

The default in PyTorch `align_corners=None`. **TODO**

## Activation Functions

PyTorch treats activation functions as steps in a sequential model, while Flux treats them as functions to be passed to a layer.
For example:

```python
nn.Sequential(
    nn.Conv2d(2, 2, 3, padding = 1),
    nn.ReLU()
)
```

```julia
CrossCorr((3, 3), 2 => 2, relu; pad = 1)
```

## Weights Initialization

PyTorch by default uses `kaiming_uniform` (aka He initialization) for weights. Flux by default uses `glorot_uniform` (aka Xavier
initialization) for weights. To use `kaiming_uniform`, pass `Flux.kaiming_uniform` to the `init` keyword argument of a layer.

Flux defaults `kaiming_uniform` to `gain = √2`. PyTorch auto-selects a `gain` for the init function based on the type of
nonlinearity specified, which by default is `leaky_relu`. It is calculated as:
```
a = √5 # argument passed into PyTorch's `kaiming_uniform_` function, the "negative slope" of the rectifier
gain = √(2 / (1 + a ^ 2))
```
If `relu` nonlinearity is specified in PyTorch weights init, its default gain matches Flux.

To replicate the PyTorch default, you can use `init = Flux.kaiming_uniform(gain = √(2 / (1 + a ^ 2)))` in Flux.

## Bias Initialization

PyTorch initializes bias parameters with uniformly random values between `± 1 / √(fan_in)`, where `fan_in` in Flux is
`first(nfan(filter..., cin÷groups, cout))` for `Conv`/`CrossCorr` layers. For `Dense` layers, `last(nfan(out, in))` instead.
Flux initializes them all to zero.

To replicate the PyTorch default, you'll need to manually initialize a vector as described and pass it to the `bias` keyword
argument of a layer.

## Forward Pass

In PyTorch, the forward pass is defined in a `forward` method of a `nn.Module` subclass. In Flux, the forward pass is defined by
creating a function on the layer object itself:

```python
class MyLayer(nn.Module):
    def forward(self, input):
        ...
```

```julia
struct MyLayer
end
(::MyLayer)(input) = ...
```

## Gradient Modification

`torch.nn.utils.clip_grad_norm_` norms the gradients across the whole network and clips them. It does not currently have a direct
equivalent in Flux. `ClipNorm` only operates within a single array. You can accomplish this with something like:
```
function global_grad_norm(grads)
    acc = 0.0
    for g in Optimisers.trainables(grads)
        g === nothing && continue
        acc += Float64(sum(abs2, g))
    end
    return sqrt(acc)
end

function clip_global_norm(grads, max_norm::Real)
    total = global_grad_norm(grads)
    coef = min(max_norm / (total + 1e-6), 1.0)
    scaled = Optimisers.fmap(grads) do g
        g isa AbstractArray{<:AbstractFloat} ? g .* eltype(g)(coef) : g
    end
    return scaled, total
end
```

## Excluding Parameters from Gradients

In PyTorch, a parameter can be excluded from gradients by registering it with `requires_grad=False`. In Flux, you can accomplish
this by annotating a function that returns the parameter with `ChainRulesCore.@non_differentiable`, and then using that function
wherever you would use that parameter. 

## Training Loop

A single step in Flux is simple `gradient` followed by `update!`. Both steps can be combined with the `train!` function, and can
also be used to iterate over a set of paired training inputs and outputs.

In PyTorch there are more steps: the optimizer's gradients must be zeroed with `.zero_grad()`, then the loss is calculated,
then the tensor returned from the loss function is backward-propagated with `.backward()` to compute the gradients, and finally
the optimizer is stepped forward and the model parameters are updated with `.step()`.

In Flux, an optimizer state object is first obtained by `setup`, and this state is passed to the training loop. In PyTorch, the
optimizer object itself is manipulated in the training loop.

## Miscellaneous

`torch.where` => `Base.ifelse.()`

`torch.squeeze` => `Base.dropdims`

Many PyTorch utility methods (e.g. `unsqueeze`) have equivalents in `MLUtils.jl`.

PyTorch has adopted the convention of using a trailing underscore to indicate a function that mutates its arguments.

PyTorch's distributions are specified with a rate, while Julia's `Distributions` uses a scale argument, where `rate = 1 / scale`.
