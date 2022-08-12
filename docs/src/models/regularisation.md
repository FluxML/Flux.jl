# Regularisation

Applying regularisation to model parameters is straightforward. We just need to
apply an appropriate regulariser to each model parameter and
add the result to the overall loss.

For example, say we have a simple regression.

```jldoctest regularisation
julia> using Flux

julia> using Flux.Losses: logitcrossentropy

julia> m = Dense(10 => 5)
Dense(10 => 5)      # 55 parameters

julia> loss(x, y) = logitcrossentropy(m(x), y);
```

We can apply L2 regularisation by taking the squared norm of the parameters , `m.weight` and `m.bias`.

```jldoctest regularisation
julia> penalty() = sum(abs2, m.weight) + sum(abs2, m.bias);

julia> loss(x, y) = logitcrossentropy(m(x), y) + penalty();
```

When working with layers, Flux provides the `params` function to grab all
parameters at once. We can easily penalise everything with `sum`:

```jldoctest regularisation; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> Flux.params(m)
Params([Float32[0.34704182 -0.48532376 … -0.06914271 -0.38398427; 0.5201164 -0.033709668 … -0.36169025 -0.5552353; … ; 0.46534058 0.17114447 … -0.4809643 0.04993277; -0.47049698 -0.6206029 … -0.3092334 -0.47857067], Float32[0.0, 0.0, 0.0, 0.0, 0.0]])

julia> sqnorm(x) = sum(abs2, x);

julia> sum(sqnorm, Flux.params(m))
8.34994f0
```

Here's a larger example with a multi-layer perceptron.

```jldoctest regularisation; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> m = Chain(Dense(28^2 => 128, relu), Dense(128 => 32, relu), Dense(32 => 10))
Chain(
  Dense(784 => 128, relu),              # 100_480 parameters
  Dense(128 => 32, relu),               # 4_128 parameters
  Dense(32 => 10),                      # 330 parameters
)                   # Total: 6 arrays, 104_938 parameters, 410.289 KiB.

julia> sqnorm(x) = sum(abs2, x);

julia> loss(x, y) = logitcrossentropy(m(x), y) + sum(sqnorm, Flux.params(m));

julia> loss(rand(28^2), rand(10))
300.76693683244997
```

One can also easily add per-layer regularisation via the `activations` function:

```jldoctest regularisation; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> using Flux: activations

julia> c = Chain(Dense(10 => 5, σ), Dense(5 => 2), softmax)
Chain(
  Dense(10 => 5, σ),                    # 55 parameters
  Dense(5 => 2),                        # 12 parameters
  NNlib.softmax,
)                   # Total: 4 arrays, 67 parameters, 524 bytes.

julia> activations(c, rand(10))
([0.3274892431795043, 0.5360197770386552, 0.3447464835514667, 0.5273025865532305, 0.7513168089280781], [-0.3533774181890544, -0.010937055274926138], [0.4152168057978045, 0.5847831942021956])

julia> sum(sqnorm, ans)
1.9953131077618562
```

```@docs
Flux.activations
```
