
"""
    nfan(n_out, n_in=1) -> Tuple
    nfan(dims...)
    nfan(dims::Tuple)

For a layer characterized by dimensions `dims`, return a tuple `(fan_in, fan_out)`, where `fan_in`
is the number of input neurons connected to an output one, and `fan_out` is the number of output neurons
connected to an input one.

This function is mainly used by weight initializers, e.g., [`kaiming_normal`](@ref Flux.kaiming_normal).

# Examples

```jldoctest
julia> layer = Dense(10, 20);

julia> Flux.nfan(size(layer.weight))
(10, 20)

julia> layer = Conv((3, 3), 2=>10);

julia> Flux.nfan(size(layer.weight))
(18, 90)
```
"""
nfan() = 1, 1 # fan_in, fan_out
nfan(n) = 1, n # A vector is treated as a n×1 matrix
nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices
nfan(dims::Tuple) = nfan(dims...)
nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end]) # In case of convolution kernels

ofeltype(x, y) = convert(float(eltype(x)), y)
epseltype(x) = eps(float(eltype(x)))

"""
    rng_from_array([x])

Create an instance of the RNG most appropriate for `x`.
The current defaults are:
- `x isa CuArray`: `CUDA.default_rng()`, else:
- `x isa AbstractArray`, or no `x` provided:
  - Julia version is < 1.7: `Random.GLOBAL_RNG`
  - Julia version is >= 1.7: `Random.default_rng()`
"""
rng_from_array(::AbstractArray) = default_rng_value()
rng_from_array(::CuArray) = CUDA.default_rng()

@non_differentiable rng_from_array(::Any)

if VERSION >= v"1.7"
  default_rng_value() = Random.default_rng()
else
  default_rng_value() = Random.GLOBAL_RNG
end

"""
    default_rng_value()

Create an instance of the default RNG depending on Julia's version.
- Julia version is < 1.7: `Random.GLOBAL_RNG`
- Julia version is >= 1.7: `Random.default_rng()`
"""
default_rng_value

"""
    glorot_uniform([rng = default_rng_value()], size...; gain = 1) -> Array
    glorot_uniform([rng]; kw...) -> Function

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a uniform
distribution on the interval ``[-x, x]``, where `x = gain * sqrt(6 / (fan_in + fan_out))`.

This method is described in [1] and also known as Xavier initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.glorot_uniform(3, 4) |> summary
"3×4 Matrix{Float32}"

julia> round.(extrema(Flux.glorot_uniform(10, 100)), digits=3)
(-0.232f0, 0.234f0)

julia> round.(extrema(Flux.glorot_uniform(100, 10)), digits=3)
(-0.233f0, 0.233f0)

julia> round.(extrema(Flux.glorot_uniform(100, 100)), digits=3)
(-0.173f0, 0.173f0)

julia> Dense(3 => 2, tanh; init = Flux.glorot_uniform(MersenneTwister(1)))
Dense(3 => 2, tanh)  # 8 parameters

julia> ans.bias
2-element Vector{Float32}:
 0.0
 0.0
```

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." _Proceedings of the thirteenth international conference on artificial intelligence and statistics_. 2010.
"""
function glorot_uniform(rng::AbstractRNG, dims::Integer...; gain::Real=1)
  scale = Float32(gain) * sqrt(24.0f0 / sum(nfan(dims...)))
  (rand(rng, Float32, dims...) .- 0.5f0) .* scale
end
glorot_uniform(dims::Integer...; kw...) = glorot_uniform(default_rng_value(), dims...; kw...)
glorot_uniform(rng::AbstractRNG=default_rng_value(); init_kwargs...) = (dims...; kwargs...) -> glorot_uniform(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable glorot_uniform(::Any...)

"""
    glorot_normal([rng = default_rng_value(), size...; gain = 1) -> Array
    glorot_normal([rng]; kw...) -> Function

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a normal
distribution with standard deviation `gain * sqrt(2 / (fan_in + fan_out))`,
using [`nfan`](@ref Flux.nfan).

This method is described in [1] and also known as Xavier initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> using Statistics

julia> round(std(Flux.glorot_normal(10, 1000)), digits=3)
0.044f0

julia> round(std(Flux.glorot_normal(1000, 10)), digits=3)
0.044f0

julia> round(std(Flux.glorot_normal(1000, 1000)), digits=3)
0.032f0

julia> Dense(10 => 1000, tanh; init = Flux.glorot_normal(gain=100))
Dense(10 => 1000, tanh)  # 11_000 parameters

julia> round(std(ans.weight), sigdigits=3)
4.45f0
```

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." _Proceedings of the thirteenth international conference on artificial intelligence and statistics_. 2010.
"""
function glorot_normal(rng::AbstractRNG, dims::Integer...; gain::Real=1)
  std = Float32(gain) * sqrt(2.0f0 / sum(nfan(dims...)))
  randn(rng, Float32, dims...) .* std
end
glorot_normal(dims::Integer...; kwargs...) = glorot_normal(default_rng_value(), dims...; kwargs...)
glorot_normal(rng::AbstractRNG=default_rng_value(); init_kwargs...) = (dims...; kwargs...) -> glorot_normal(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable glorot_normal(::Any...)

"""
    kaiming_uniform([rng = default_rng_value()], size...; gain = √2) -> Array
    kaiming_uniform([rng]; kw...) -> Function

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a uniform distribution
on the interval `[-x, x]`, where `x = gain * sqrt(3/fan_in)` using [`nfan`](@ref Flux.nfan).

This method is described in [1] and also known as He initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> round.(extrema(Flux.kaiming_uniform(100, 10)), digits=3)
(-0.774f0, 0.774f0)

julia> round.(extrema(Flux.kaiming_uniform(10, 100)), digits=3)
(-0.245f0, 0.244f0)

julia> round.(extrema(Flux.kaiming_uniform(100, 100)), digits=3)
(-0.245f0, 0.245f0)
```

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." _Proceedings of the IEEE international conference on computer vision_. 2015.
"""
function kaiming_uniform(rng::AbstractRNG, dims::Integer...; gain::Real = √2)
  bound = Float32(√3 * gain / sqrt(first(nfan(dims...)))) # fan_in
  return (rand(rng, Float32, dims...) .- 0.5f0) .* 2bound
end

kaiming_uniform(dims::Integer...; kwargs...) = kaiming_uniform(default_rng_value(), dims...; kwargs...)
kaiming_uniform(rng::AbstractRNG=default_rng_value(); init_kwargs...) = (dims...; kwargs...) -> kaiming_uniform(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable kaiming_uniform(::Any...)

"""
    kaiming_normal([rng = default_rng_value()], size...; gain = √2) -> Array
    kaiming_normal([rng]; kw...) -> Function

Return an `Array{Float32}` of the given `size` containing random numbers taken from a normal
distribution standard deviation `gain / sqrt(fan_in)`, using [`nfan`](@ref Flux.nfan).

This method is described in [1] and also known as He initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> using Statistics

julia> round(std(Flux.kaiming_normal(10, 1000)), digits=3)
0.045f0

julia> round(std(Flux.kaiming_normal(1000, 10)), digits=3)
0.447f0

julia> round(std(Flux.kaiming_normal(1000, 1000)), digits=3)
0.045f0
```

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." _Proceedings of the IEEE international conference on computer vision_. 2015.
"""
function kaiming_normal(rng::AbstractRNG, dims::Integer...; gain::Real = √2f0)
  std = Float32(gain / sqrt(first(nfan(dims...)))) # fan_in
  return randn(rng, Float32, dims...) .* std
end

kaiming_normal(dims::Integer...; kwargs...) = kaiming_normal(default_rng_value(), dims...; kwargs...)
kaiming_normal(rng::AbstractRNG; init_kwargs...) = (dims...; kwargs...) -> kaiming_normal(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable kaiming_normal(::Any...)

"""
    truncated_normal([rng = default_rng_value()], size...; mean = 0, std = 1, lo = -2, hi = 2) -> Array
    truncated_normal([rng]; kw...) -> Function
  
Return an `Array{Float32}` of the given `size` where each element is drawn from a truncated normal distribution.
The numbers are distributed like `filter(x -> lo<=x<=hi, mean .+ std .* randn(100))`.

The values are generated by sampling a Uniform(0, 1) (`rand()`) and then
applying the inverse CDF of the truncated normal distribution.
This method works best when `lo ≤ mean ≤ hi`.

# Examples
```jldoctest
julia> using Statistics

julia> Flux.truncated_normal(3, 4) |> summary
"3×4 Matrix{Float32}"

julia> round.(extrema(Flux.truncated_normal(10^6)); digits=3)
(-2.0f0, 2.0f0)

julia> round(std(Flux.truncated_normal(10^6; lo = -100, hi = 100)))
1.0f0
```
"""
function truncated_normal(rng::AbstractRNG, dims::Integer...; mean = 0, std = 1, lo = -2, hi = 2)
  norm_cdf(x) = 0.5 * (1 + erf(x/√2))
  if (mean < lo - 2 * std) || (mean > hi + 2 * std)
    @warn "Mean is more than 2 std outside the limits in truncated_normal, so the distribution of values may be inaccurate." maxlog=1
  end
  l = norm_cdf((lo - mean) / std)
  u = norm_cdf((hi - mean) / std)
  xs = rand(rng, Float32, dims...)
  broadcast!(xs, xs) do x
    x = x * 2(u - l) + (2l - 1)
    x = erfinv(x)
    x = clamp(x * std * √2 + mean, lo, hi)
  end
  return xs
end

truncated_normal(dims::Integer...; kwargs...) = truncated_normal(default_rng_value(), dims...; kwargs...)
truncated_normal(rng::AbstractRNG=default_rng_value(); init_kwargs...) = (dims...; kwargs...) -> truncated_normal(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable truncated_normal(::Any...)

"""
    orthogonal([rng = default_rng_value()], size...; gain = 1) -> Array
    orthogonal([rng]; kw...) -> Function

Return an `Array{Float32}` of the given `size` which is a (semi) orthogonal matrix, as described in [1].

Cannot construct a vector, i.e. `length(size) == 1` is forbidden.
For `length(size) > 2`, a `prod(size[1:(end - 1)])` by `size[end]` orthogonal matrix
is computed before reshaping it to the original dimensions.

# Examples
```jldoctest; setup = :(using LinearAlgebra)
julia> W = Flux.orthogonal(5, 7);

julia> summary(W)
"5×7 Matrix{Float32}"

julia> W * W' ≈ I(5)
true

julia> W2 = Flux.orthogonal(7, 5);

julia> W2 * W2' ≈ I(7)
false

julia> W2' * W2 ≈ I(5)
true

julia> W3 = Flux.orthogonal(3, 3, 2, 4);

julia> transpose(reshape(W3, :, 4)) * reshape(W3, :, 4) ≈ I(4)
true
```

# References

[1] Saxe, McClelland, Ganguli. "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks", ICLR 2014, https://arxiv.org/abs/1312.6120

"""
function orthogonal(rng::AbstractRNG, rows::Integer, cols::Integer; gain::Real = 1)
  if rows < cols
    return permutedims(orthogonal(rng, cols, rows; gain))
  end
  mat = randn(rng, Float32, rows, cols)
  Q, R = LinearAlgebra.qr(mat)
  mat .= Array(Q) * sign.(LinearAlgebra.Diagonal(R)) .* Float32(gain)
  return mat
end

function orthogonal(rng::AbstractRNG, d1::Integer, ds::Integer...; kwargs...)
  dims = (d1, ds...)
  rows = prod(dims[1:end-1])
  cols = dims[end]
  return reshape(orthogonal(rng, rows, cols; kwargs...), dims)
end

orthogonal(dims::Integer...; kwargs...) = orthogonal(default_rng_value(), dims...; kwargs...)
orthogonal(rng::AbstractRNG=default_rng_value(); init_kwargs...) = (dims::Integer...; kwargs...) -> orthogonal(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable orthogonal(::Any...)

"""
    sparse_init([rng = default_rng_value()], rows, cols; sparsity, std = 0.01) -> Array
    sparse_init([rng]; kw...) -> Function

Return a `Matrix{Float32}` of size `rows, cols` where each column contains a fixed fraction of
zero elements given by `sparsity`. Non-zero elements are normally distributed
with a mean of zero and standard deviation `std`.

This method is described in [1].

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> count(iszero, Flux.sparse_init(10, 10, sparsity=1/5))
20

julia> sum(0 .== Flux.sparse_init(10, 11, sparsity=0.9), dims=1)
1×11 Matrix{Int64}:
 9  9  9  9  9  9  9  9  9  9  9

julia> Dense(3 => 10, tanh; init=Flux.sparse_init(sparsity=0.5))
Dense(3 => 10, tanh)  # 40 parameters

julia> count(iszero, ans.weight, dims=1)
1×3 Matrix{Int64}:
 5  5  5
```

# References

[1] Martens, J, "Deep learning via Hessian-free optimization" _Proceedings of the 27th International Conference on International Conference on Machine Learning_. 2010.
"""
function sparse_init(rng::AbstractRNG, dims::Integer...; sparsity, std = 0.01)
  if length(dims) != 2
    throw(ArgumentError("Only 2-dimensional outputs are supported for sparse initialization."))
  end
  rows, cols = dims
  prop_zero = min(1.0, sparsity)
  num_zeros = ceil(Integer, prop_zero * rows)
  sparse_array = randn(rng, Float32, dims...) .* Float32(std)
  sparse_array[1:num_zeros, :] .= 0f0
  return mapslices(shuffle, sparse_array, dims=1)
end

sparse_init(dims::Integer...; kwargs...) = sparse_init(default_rng_value(), dims...; kwargs...)
sparse_init(rng::AbstractRNG=default_rng_value(); init_kwargs...) = (dims...; kwargs...) -> sparse_init(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable sparse_init(::Any...)

"""
    identity_init(size...; gain=1, shift=0) -> Array
    identity_init(; kw...) -> Function

Return an `Array{Float32}` of the given `size` which yields an identity mapping when used as parameters in
most Flux layers. Use `gain` to scale the identity by a constant.

Often useful in the context of transfer learning, i.e when one wants to add more capacity to
a model but start from the same mapping.

Has the following behaviour
*  1D: A `Vector` of `zeros` (useful for an identity bias)
*  2D: An identity matrix (useful for an identity matrix multiplication)
*  More than 2D: A dense block array of center tap spatial filters (useful for an identity convolution)

Some caveats: 
* Not all layers will be identity mapping when used with this init. Exceptions
  include recurrent layers and normalization layers.

* Layers must have `input_size == output_size` for identity mapping to be
  possible. When this is not the case, extra dimensions of the array are padded with zeros.

* For convolutional layers, in addition to the above, the kernel sizes must also be odd and
  padding must be applied so that output feature maps have the same size as input feature maps,
  e.g by using [`SamePad`](@ref).

Use keyword `shift` (integer or tuple) to apply circular shift to the output,
equivalent to `Base.circshift(identity_init(size...), shift)`.

For consistency with other initialisers, it accepts `rng::AbstractRNG` as an optional
first argument. But this is ignored, since the result is not random.

# Examples
```jldoctest
julia> Flux.identity_init(3,5)
3×5 Matrix{Float32}:
 1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0

julia> Dense(5 => 3, relu, init=Flux.identity_init)([1,-2,3,-4,5])
3-element Vector{Float32}:
 1.0
 0.0
 3.0

julia> Flux.identity_init(3,3,2; gain=100)
3×3×2 Array{Float32, 3}:
[:, :, 1] =
   0.0  0.0  0.0
 100.0  0.0  0.0
   0.0  0.0  0.0

[:, :, 2] =
 0.0    0.0  0.0
 0.0  100.0  0.0
 0.0    0.0  0.0

julia> x4 = cat([1 2 3; 4 5 6; 7 8 9]; dims=4);

julia> Conv((2,2), 1 => 1, init=Flux.identity_init(gain=10), pad=SamePad())(x4)
3×3×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 10.0  20.0  30.0
 40.0  50.0  60.0
 70.0  80.0  90.0
```
"""
identity_init(cols::Integer; gain::Real=1, shift=0) = zeros32(cols) # Assume bias

# Assume matrix multiplication
identity_init(rows::Integer, cols::Integer; gain::Real=1, shift=0) = circshift(Matrix{Float32}(I * gain, rows,cols), shift)

# Assume convolution
function identity_init(dims::Integer...; gain::Real=1, shift=0)
  nin, nout = dims[end-1], dims[end]
  centers = map(d -> cld(d, 2), dims[1:end-2])
  weights = zeros32(dims...)
  for i in 1:min(nin,nout)
    weights[centers..., i, i] = gain
  end
  return circshift(weights, shift)
end

# For consistency, it accepts an RNG, but ignores it:
identity_init(::AbstractRNG, dims::Integer...; kwargs...) = identity_init(dims...; kwargs...)
identity_init(rng::AbstractRNG=default_rng_value(); init_kwargs...) = (args...;kwargs...) -> identity_init(rng, args...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable identity_init(::Any...)

"""
    ones32(size...) = ones(Float32, size...)

Return an `Array{Float32}` of the given `size` filled with 1s.
"""
ones32(dims...) = Base.ones(Float32, dims...)

"""
    zeros32(size...) = zeros(Float32, size...)

Return an `Array{Float32}` of the given `size` filled with 0s.
"""
zeros32(dims...) = Base.zeros(Float32, dims...)

"""
    rand32([rng], size...)

Return an `Array{Float32}` of the given `size`, filled like `rand`.
When the size is not provided, `rand32(rng::AbstractRNG)` returns a function.
"""
rand32(dims::Integer...) = Base.rand(Float32, dims...)
rand32(rng::AbstractRNG, dims::Integer...) = Base.rand(rng, Float32, dims...)
rand32(rng::AbstractRNG) = (dims...,) -> Base.rand(rng, Float32, dims...)

"""
    randn32([rng], size...)

Return an `Array{Float32}` of the given `size`, filled like `randn`.
When the size is not provided, `randn32(rng::AbstractRNG)` returns a function.
"""
randn32(dims::Integer...) = Base.randn(Float32, dims...)
randn32(rng::AbstractRNG, dims::Integer...) = Base.randn(rng, Float32, dims...)
randn32(rng::AbstractRNG) = (dims...,) -> Base.randn(rng, Float32, dims...)

"""
    create_bias(weights, bias, size...)

Return a bias parameter for a layer, based on the value given
to the constructor's keyword `bias=bias`.

* `bias == true` creates a trainable array of the given size, of the same type as `weights`, initialised to zero.
* `bias == false` returns `false`, which is understood by AD to be non-differentiable.
* `bias::AbstractArray` uses the array provided, provided it has the correct size.
  It will also correct the `eltype` to match that of `weights`.
"""
function create_bias(weights::AbstractArray, bias::Bool, dims::Integer...)
  bias ? fill!(similar(weights, dims...), 0) : false
end
function create_bias(weights::AbstractArray, bias::AbstractArray, dims::Integer...)
  size(bias) == dims || throw(DimensionMismatch("expected bias of size $(dims), got size $(size(bias))"))
  convert(AbstractArray{eltype(weights)}, bias)
end


# Other

"""
    throttle(f, timeout; leading=true, trailing=false)

Return a function that when invoked, will only be triggered at most once
during `timeout` seconds.

Normally, the throttled function will run as much as it can, without ever
going more than once per `wait` duration; but if you'd like to disable the
execution on the leading edge, pass `leading=false`. To enable execution on
the trailing edge, pass `trailing=true`.

# Examples
```jldoctest
julia> a = Flux.throttle(() -> println("Flux"), 2);

julia> for i = 1:4  # a called in alternate iterations
           a()
           sleep(1)
       end
Flux
Flux
```
"""
function throttle(f, timeout; leading=true, trailing=false)
  cooldown = true
  later = nothing
  result = nothing

  function throttled(args...; kwargs...)
    yield()

    if cooldown
      if leading
        result = f(args...; kwargs...)
      else
        later = () -> f(args...; kwargs...)
      end

      cooldown = false
      @async try
        while (sleep(timeout); later != nothing)
          later()
          later = nothing
        end
      finally
        cooldown = true
      end
    elseif trailing
      later = () -> (result = f(args...; kwargs...))
    end

    return result
  end
end


"""
    modules(m)

Return an iterator over non-leaf objects
that can be reached by recursing `m` over
the children given by [`functor`](@ref).

Useful for applying a function (e.g. a regularizer)
over specific modules or subsets of the parameters
(e.g. the weights but not the biases).

# Examples

```jldoctest
julia> m1 = Chain(Dense(28^2, 64), BatchNorm(64, relu));

julia> m2 = Chain(m1, Dense(64, 10))
Chain(
  Chain(
    Dense(784 => 64),                   # 50_240 parameters
    BatchNorm(64, relu),                # 128 parameters, plus 128
  ),
  Dense(64 => 10),                      # 650 parameters
)         # Total: 6 trainable arrays, 51_018 parameters,
          # plus 2 non-trainable, 128 parameters, summarysize 200.312 KiB.

julia> Flux.modules(m2)
7-element Vector{Any}:
 Chain(Chain(Dense(784 => 64), BatchNorm(64, relu)), Dense(64 => 10))  # 51_018 parameters, plus 128 non-trainable
 (Chain(Dense(784 => 64), BatchNorm(64, relu)), Dense(64 => 10))
 Chain(Dense(784 => 64), BatchNorm(64, relu))  # 50_368 parameters, plus 128 non-trainable
 (Dense(784 => 64), BatchNorm(64, relu))
 Dense(784 => 64)    # 50_240 parameters
 BatchNorm(64, relu)  # 128 parameters, plus 128 non-trainable
 Dense(64 => 10)     # 650 parameters

julia> L2(m) = sum(sum(abs2, l.weight) for l in Flux.modules(m) if l isa Dense)
L2 (generic function with 1 method)

julia> L2(m2) isa Float32
true
```
"""
modules(m) = [x for x in Functors.fcollect(m) if !isleaflike(x)]

@nograd modules # TODO: is this correct? might fail with explicit parameters.
function ChainRulesCore.rrule(::typeof(modules), m)
  modules(m), dm -> error("Flux.modules is not at present differentiable, sorry")
end

isleaflike(x) = Functors.isleaf(x)
isleaflike(::Tuple{Vararg{<:Number}}) = true
isleaflike(::Tuple{Vararg{<:AbstractArray{<:Number}}}) = true

"""
    patience(predicate, wait)

Return a function that internally counts by one when
`predicate(...) == true`, otherwise the count is reset to zero.
If the count is greater than or equal to `wait`,
the function returns `true`, otherwise it returns `false`.

# Examples
```jldoctest
julia> loss() = rand();

julia> trigger = Flux.patience(() -> loss() < 1, 3);


julia> for i in 1:10
         @info "Epoch \$i"
         trigger() && break
       end
[ Info: Epoch 1
[ Info: Epoch 2
[ Info: Epoch 3
```
"""
function patience(predicate, wait)
  let count = 0
    function on_trigger(args...; kwargs...)
      count = predicate(args...; kwargs...) ? count + 1 : 0

      return count >= wait
    end
  end
end

"""
    early_stopping(f, delay; distance = -, init_score = 0, min_dist = 0)

Return a function that internally counts by one when
`distance(best_score, f(...)) <= min_dist`, where
`best_score` is the last seen best value of `f(...)`.
If the count is greater than or equal to `delay`,
the function returns `true`, otherwise it returns `false`.
The count is reset when `distance(best_score, f(...)) > min_dist`.

# Examples
```jldoctest
julia> loss = let l = 0
         () -> l += 1
       end; # pseudo loss function that returns increasing values

julia> es = Flux.early_stopping(loss, 3);


julia> for i in 1:10
         @info "Epoch \$i"
         es() && break
       end
[ Info: Epoch 1
[ Info: Epoch 2
[ Info: Epoch 3
```
"""
function early_stopping(f, delay; distance = -, init_score = 0, min_dist = 0)
  trigger = let best_score = init_score
    (args...; kwargs...) -> begin
      score = f(args...; kwargs...)
      Δ = distance(best_score, score)
      best_score = Δ < 0 ? best_score : score

      return Δ < min_dist
    end
  end

  return patience(trigger, delay)
end

"""
    plateau(f, width; distance = -, init_score = 0, min_dist = 1f-6)

Return a function that internally counts by one when
`abs(distance(last_score, f(...))) <= min_dist`, where
`last_score` holds the last value of `f(...)`.
If the count is greater than or equal to `width`,
the function returns `true`, otherwise it returns `false`.
The count is reset when `abs(distance(last_score, f(...))) > min_dist`.

# Examples
```jldoctest
julia> f = let v = 10
         () -> v = v / abs(v) - v
       end; # -9, 8, -7, 6, ...

julia> trigger = Flux.plateau(f, 3; init_score=10, min_dist=18);


julia> for i in 1:10
         @info "Epoch \$i"
         trigger() && break
       end
[ Info: Epoch 1
[ Info: Epoch 2
[ Info: Epoch 3
[ Info: Epoch 4
```
"""
function plateau(f, width; distance = -, init_score = 0, min_dist = 1f-6)
  is_plateau = let last_score = init_score
    (args...; kwargs...) -> begin
      score = f(args...; kwargs...)
      Δ = abs(distance(last_score, score))
      last_score = score

      return Δ < min_dist
    end
  end

  return patience(is_plateau, width)
end
