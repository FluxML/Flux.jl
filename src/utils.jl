# Arrays
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
- `x isa AbstractArray`
  - Julia version is < 1.7: `Random.GLOBAL_RNG`
  - Julia version is >= 1.7: `Random.default_rng()`
- `x isa CuArray`: `CUDA.default_rng()`
When `x` is unspecified, it is assumed to be a `AbstractArray`.
"""
rng_from_array(::AbstractArray) = rng_from_array()
rng_from_array(::CuArray) = CUDA.default_rng()
if VERSION >= v"1.7"
  rng_from_array() = Random.default_rng()
else
  rng_from_array() = Random.GLOBAL_RNG
end

"""
    glorot_uniform([rng=GLOBAL_RNG], dims...)

Return an `Array` of size `dims` containing random variables taken from a uniform
distribution in the interval ``[-x, x]``, where `x = sqrt(6 / (fan_in + fan_out))`.

This method is described in [1] and also known as Xavier initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.glorot_uniform(2, 3)
2×3 Matrix{Float32}:
 0.601094  -0.57414   -0.814925
 0.900868   0.805994   0.057514
```

# See also

* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* sparse initialization: [`sparse_init`](@ref Flux.sparse_init)
* calculation of `fan_in` and `fan_out`: [`nfan`](@ref Flux.nfan)

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." _Proceedings of the thirteenth international conference on artificial intelligence and statistics_. 2010.
"""
glorot_uniform(rng::AbstractRNG, dims::Integer...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum(nfan(dims...)))
glorot_uniform(dims::Integer...) = glorot_uniform(rng_from_array(), dims...)
glorot_uniform(rng::AbstractRNG) = (dims...) -> glorot_uniform(rng, dims...)

ChainRulesCore.@non_differentiable glorot_uniform(::Any...)

"""
    glorot_normal([rng=GLOBAL_RNG], dims...)

Return an `Array` of size `dims` containing random variables taken from a normal
distribution with mean 0 and standard deviation `sqrt(2 / (fan_in + fan_out))`.

This method is described in [1] and also known as Xavier initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.glorot_normal(3, 2)
3×2 Matrix{Float32}:
  0.429505  -0.0852891
  0.523935   0.371009
 -0.223261   0.188052
```

# See also

* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)
* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* sparse initialization: [`sparse_init`](@ref Flux.sparse_init)
* calculation of `fan_in` and `fan_out`: [`nfan`](@ref Flux.nfan)

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." _Proceedings of the thirteenth international conference on artificial intelligence and statistics_. 2010.
"""
glorot_normal(rng::AbstractRNG, dims::Integer...) = randn(rng, Float32, dims...) .* sqrt(2.0f0 / sum(nfan(dims...)))
glorot_normal(dims::Integer...) = glorot_normal(rng_from_array(), dims...)
glorot_normal(rng::AbstractRNG) = (dims...) -> glorot_normal(rng, dims...)

ChainRulesCore.@non_differentiable glorot_normal(::Any...)

"""
    kaiming_uniform([rng=GLOBAL_RNG], dims...; gain = √2)

Return an `Array` of size `dims` containing random variables taken from a uniform distribution in the
interval `[-x, x]`, where `x = gain * sqrt(3/fan_in)`.

This method is described in [1] and also known as He initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.kaiming_uniform(3, 2)
3×2 Matrix{Float32}:
  0.950413   1.27439
  1.4244    -1.28851
 -0.907795   0.0909376
```

# See also

* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)
* sparse initialization: [`sparse_init`](@ref Flux.sparse_init)
* calculation of `fan_in` and `fan_out`: [`nfan`](@ref Flux.nfan)

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." _Proceedings of the IEEE international conference on computer vision_. 2015.
"""
function kaiming_uniform(rng::AbstractRNG, dims::Integer...; gain = √2)
  bound = Float32(√3 * gain / sqrt(first(nfan(dims...)))) # fan_in
  return (rand(rng, Float32, dims...) .- 0.5f0) .* 2bound
end

kaiming_uniform(dims::Integer...; kwargs...) = kaiming_uniform(rng_from_array(), dims...; kwargs...)
kaiming_uniform(rng::AbstractRNG; init_kwargs...) = (dims...; kwargs...) -> kaiming_uniform(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable kaiming_uniform(::Any...)

"""
    kaiming_normal([rng=GLOBAL_RNG], dims...; gain = √2)

Return an `Array` of size `dims` containing random variables taken from a normal
distribution with mean 0 and standard deviation `gain * sqrt(fan_in)`.

This method is described in [1] and also known as He initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.kaiming_normal(3, 2)
3×2 Matrix{Float32}:
  0.679107  -0.134854
  0.828413   0.586617
 -0.353007   0.297336
```

# See also

* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)
* sparse initialization: [`sparse_init`](@ref Flux.sparse_init)
* calculation of `fan_in` and `fan_out`: [`nfan`](@ref Flux.nfan)

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." _Proceedings of the IEEE international conference on computer vision_. 2015.
"""
function kaiming_normal(rng::AbstractRNG, dims::Integer...; gain = √2f0)
  std = Float32(gain / sqrt(first(nfan(dims...)))) # fan_in
  return randn(rng, Float32, dims...) .* std
end

kaiming_normal(dims::Integer...; kwargs...) = kaiming_normal(rng_from_array(), dims...; kwargs...)
kaiming_normal(rng::AbstractRNG; init_kwargs...) = (dims...; kwargs...) -> kaiming_normal(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable kaiming_normal(::Any...)

"""
    truncated_normal([rng=GLOBAL_RNG], dims...; mean = 0, std = 1, lo = -2, hi = 2)
  
Return an `Array{Float32}` of size `dims` where each element is drawn from a truncated normal distribution.
The numbers are distributed like `filter(x -> lo<=x<=hi, mean .+ std .* randn(dims...))`.

The values are generated by sampling a Uniform(0, 1) (`rand()`) and then
applying the inverse CDF of the truncated normal distribution
(see the references for more info).
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

# References
[1] Burkardt, John. "The Truncated Normal Distribution" 
[PDF](https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf). 
Department of Scientific Computing website.
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

truncated_normal(dims::Integer...; kwargs...) = truncated_normal(rng_from_array(), dims...; kwargs...)
truncated_normal(rng::AbstractRNG; init_kwargs...) = (dims...; kwargs...) -> truncated_normal(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable truncated_normal(::Any...)

"""
    orthogonal([rng=GLOBAL_RNG], dims...; gain = 1)

Return an `Array` of size `dims` which is a (semi) orthogonal matrix, as described in [1].

The input must have at least 2 dimensions.
For `length(dims) > 2`, a `prod(dims[1:(end - 1)])` by `dims[end]` orthogonal matrix
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

# See also
* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)
* sparse initialization: [`sparse_init`](@ref Flux.sparse_init)

# References

[1] Saxe, McClelland, Ganguli. "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks", ICLR 2014, https://arxiv.org/abs/1312.6120

"""
function orthogonal(rng::AbstractRNG, rows::Integer, cols::Integer; gain = 1)
  mat = rows > cols ? randn(rng, Float32, rows, cols) : randn(rng, Float32, cols, rows)

  Q, R = LinearAlgebra.qr(mat)
  Q = Array(Q) * sign.(LinearAlgebra.Diagonal(R))
  if rows < cols
    Q = transpose(Q)
  end

  return gain * Q
end

function orthogonal(rng::AbstractRNG, d1::Integer, ds::Integer...; kwargs...)
  dims = (d1, ds...)
  rows = prod(dims[1:end-1])
  cols = dims[end]
  return reshape(orthogonal(rng, rows, cols; kwargs...), dims)
end

orthogonal(dims::Integer...; kwargs...) = orthogonal(rng_from_array(), dims...; kwargs...)
orthogonal(rng::AbstractRNG; init_kwargs...) = (dims::Integer...; kwargs...) -> orthogonal(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable orthogonal(::Any...)

"""
    sparse_init([rng=GLOBAL_RNG], dims...; sparsity, std = 0.01)

Return an `Array` of size `dims` where each column contains a fixed fraction of
zero elements given by `sparsity`. Non-zero elements are normally distributed
with a mean of zero and standard deviation `std`.

This method is described in [1].

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.sparse_init(3, 2, sparsity=0.1)
3×2 Matrix{Float32}:
  0.00828413  0.0
 -0.00353007  0.00297336
  0.0         0.00586617
```

# See also

* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)

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

sparse_init(dims::Integer...; kwargs...) = sparse_init(rng_from_array(), dims...; kwargs...)
sparse_init(rng::AbstractRNG; init_kwargs...) = (dims...; kwargs...) -> sparse_init(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable sparse_init(::Any...)

"""
    identity_init([rng=GLOBAL_RNG], dims...; gain=1, shift=0)

Return an `Array` of size `dims` which yields an identity mapping when used as parameters in
most Flux layers. Use `gain` to scale the identity by a constant.

Often useful in the context of transfer learning, i.e when one wants to add more capacity to
a model but start from the same mapping.

Use `shift` (integer or tuple) to apply circular shift to the output.
Equivalent to `Base.circshift(identity(dims...), shift)`.

Some caveats: Not all layers will be identity mapping when used with this init. Exceptions
include recurrent layers, `DepthwiseConv` and normalization layers.

Also note that layers must have `input_size == output_size` for identity mapping to be
possible. When this is not the case, extra dimensions of the array are padded with zeros.

For convolutional layers, in addition to the above, the kernel sizes must also be odd and
padding must be applied so that output feature maps have the same size as input feature maps,
e.g by using [`SamePad`](@ref).

Has the following behaviour
*  1D: A `Vector` of `zeros` (useful for an identity bias)
*  2D: An identity matrix (useful for an identity matrix multiplication)
*  More than 2D: A dense block array of center tap spatial filters (useful for an identity convolution)

```jldoctest
julia> Flux.identity_init(3,3)
3×3 Matrix{Float32}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0

julia> Flux.identity_init(3,5)
3×5 Matrix{Float32}:
 1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0

julia> Flux.identity_init(3,3,2,2)
3×3×2×2 Array{Float32,4}:
[:, :, 1, 1] =
 0.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  0.0

[:, :, 2, 1] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 1, 2] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2, 2] =
 0.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  0.0
```
"""
# Assume bias
identity_init(cols::Integer; gain=1, shift=0) = zeros32(cols)

# Assume matrix multiplication
identity_init(rows::Integer, cols::Integer; gain=1, shift=0) = circshift(Matrix{Float32}(I * gain, rows,cols), shift)

# Assume convolution
function identity_init(dims::Integer...; gain=1, shift=0)
  nin, nout = dims[end-1], dims[end]
  centers = map(d -> cld(d, 2), dims[1:end-2])
  weights = zeros32(dims...)
  for i in 1:min(nin,nout)
    weights[centers..., i, i] = gain
  end
  return circshift(weights, shift)
end

identity_init(::AbstractRNG, dims::Integer...; kwargs...) = identity_init(dims...; kwargs...)
identity_init(; init_kwargs...) = identity_init(rng_from_array(); init_kwargs...)
identity_init(rng::AbstractRNG; init_kwargs...) = (args...;kwargs...) -> identity_init(rng, args...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable identity_init(::Any...)

ones32(dims::Integer...) = Base.ones(Float32, dims...)
zeros32(dims::Integer...) = Base.zeros(Float32, dims...)
rand32(dims::Integer...) = Base.rand(Float32, dims...)
randn32(dims::Integer...) = Base.randn(Float32, dims...)

"""
    create_bias(weights, bias, size...)

Return a bias parameter for a layer, based on the value given
to the constructor's keyword `bias=bias`.

* `bias == true` creates a trainable array of the given size, of the same type as `weights`, initialised to zero.
* `bias == false` returns `false`, which is understood by AD to be non-differentiable.
* `bias::AbstractArray` uses the array provided, provided it has the correct size.
  It does not at present correct the `eltype` to match that of `weights`.
"""
function create_bias(weights::AbstractArray, bias::Bool, dims::Integer...)
  bias ? fill!(similar(weights, dims...), 0) : false
end
function create_bias(weights::AbstractArray, bias::AbstractArray, dims::Integer...)
  size(bias) == dims || throw(DimensionMismatch("expected bias of size $(dims), got size $(size(bias))"))
  bias
end

# Flattening models to weight vectors, and back

function _restructure(m, xs)
  i = 0
  m̄ = fmap(m) do x
    x isa AbstractArray || return x
    x = reshape(xs[i.+(1:length(x))], size(x))
    i += length(x)
    return x
  end
  length(xs) == i || @warn "Expected $(i) params, got $(length(xs))"
  return m̄
end

@adjoint function _restructure(m, xs)  # TODO ChainRulesCore.rrule
  m̄, numel = _restructure(m, xs), length(xs)
  function _restructure_pullback(dm)
    xs′ = destructure(dm)[1]
    numel == length(xs′) || @warn "Expected $(numel) params, got $(length(xs′))"
    return (nothing, xs′)
  end
  return m̄, _restructure_pullback
end

"""
    destructure(m)

Flatten a model's parameters into a single weight vector.

    julia> m = Chain(Dense(10, 5, std), Dense(5, 2), softmax)
    Chain(Dense(10, 5, std), Dense(5, 2), softmax)

    julia> θ, re = destructure(m);

    julia> θ
    67-element Vector{Float32}:
    -0.1407104
    ...

The second return value `re` allows you to reconstruct the original network after making
modifications to the weight vector (for example, with a hypernetwork).

    julia> re(θ .* 2)
    Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
"""
function destructure(m)
  xs = Zygote.Buffer([])
  fmap(m) do x
    x isa AbstractArray && push!(xs, x)
    return x
  end
  return vcat(vec.(copy(xs))...), p -> _restructure(m, p)
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


julia> Flux.@epochs 10 begin
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


julia> Flux.@epochs 10 begin
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


julia> Flux.@epochs 10 begin
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
