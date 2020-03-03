import Adapt: adapt, adapt_storage
using Zygote: IdSet

"""
  functor(x) -> func, re

We have `x == re(func)`. 
Return `func = ()` and `re = _ -> x` for leaf objects.
"""
function functor end

# by default, every object is a leaf
functor(x) = (), _ -> x

functor(x::Tuple) = x, y -> y
functor(x::NamedTuple) = x, y -> y

functor(x::AbstractArray) = x, y -> y
functor(x::AbstractArray{<:Number}) = (), _ -> x

function makefunctor(m::Module, T, fs = fieldnames(T))
  @eval m begin
    Flux.functor(x::$T) = ($([:($f=x.$f) for f in fs]...),), y -> $T(y...)
  end
end

function functorm(T, fs = nothing)
  fs == nothing || isexpr(fs, :tuple) || error("@functor T (a, b)")
  fs = fs == nothing ? [] : [:($(map(QuoteNode, fs.args)...),)]
  :(makefunctor(@__MODULE__, $(esc(T)), $(fs...)))
end

"""
  @functor T  fields...
  
Given a type `T` and a subset of its fieldnames `fields`,
create a [`functor`](@ref) function :

  functor(x::T) -> func, re

where 

  func: (field1 = x.field1, field2 = x.field2, ....)

  re: y -> T(y...)  

If no `fields` argument is given, all internal fields will be considered. 
"""
macro functor(args...)
  functorm(args...)
end

"""
  isleaf(x)

Check if variable `x` is a *leaf* according to the definition:
    
  isleaf(x) = functor(x)[1] === ()

See [`functor`](@ref).
"""
isleaf(x) = functor(x)[1] === ()

function fmap1(f, x)
  func, re = functor(x)
  re(map(f, func))
end

"""
  fmap(f, m)
  
Applies function `f` to each leaf (see [`isleaf`](@ref)) in `m` and reconstructs 
`m` from the transformed leaves. 

Example:

  gpu(m) = fmap(CuArrays.cu, m)

"""
function fmap(f, x; cache = IdDict())
  haskey(cache, x) && return cache[x]
  cache[x] = isleaf(x) ? f(x) : fmap1(x -> fmap(f, x, cache = cache), x)
end

trainable(m) = functor(m)[1]

"""
    testmode!(m, mode = true)

Set a layer or model's test mode (see below).
Using `:auto` mode will treat any gradient computation as training.

_Note_: if you manually set a model into test mode, you need to manually place
it back into train mode during training phase.

Possible values include:
- `false` for training
- `true` for testing
- `:auto` or `nothing` for Flux to detect the mode automatically
"""
testmode!(m, mode = true) = m

"""
    trainmode!(m, mode = true)

Set a layer of model's train mode (see below).
Symmetric to [`testmode!`](@ref) (i.e. `trainmode!(m, mode) == testmode!(m, !mode)).

_Note_: if you manually set a model into train mode, you need to manually place
it into test mode during testing phase.

Possible values include:
- `true` for training
- `false` for testing
- `:auto` or `nothing` for Flux to detect the mode automatically
"""
trainmode!(m, mode = true) = mode isa Bool ? testmode!(m, !mode) : testmode!(m, mode)

params!(p::Params, x::AbstractArray{<:Number}, seen = IdSet()) = push!(p, x)

function params!(p::Params, x, seen = IdSet())
  x in seen && return
  push!(seen, x)
  for child in trainable(x)
    params!(p, child, seen)
  end
end

"""
  params(x...)

Recursively scans the inputs for trainable params
and collects them into a `Zygote.Params` object `ps`.

***Usage***

  W = rand(5, 3)
  b = zeros(5)
  m = Dense(W, b)
  
  ps = params(W, b)
  ps = params([W, b]) # equivalent form
  ps = params(m) # equivalent form

  x = rand(3)
  y = rand(5)
  loss(W, b) = sum(((W*x + b) - y).^2)
  loss(m) = sum((m(x) - y).^2)

  # Gradient computation.
  # Returns a tuple of 2 of arrays containing the gradients. 
  gs = gradient((W, b) -> loss(W, b), W, b)

  # Gradient behaves differently with Params.
  # ps is not fed as an argument to the loss.
  # Returns a Zygote.Grads object.
  gs = gradient(() -> loss(m), ps) 

"""
function params(x...)
  ps = Params()
  params!(ps, x)
  return ps
end

function loadparams!(m, xs)
  for (p, x) in zip(params(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(p, x)
  end
end

# CPU/GPU movement conveniences
"""
  cpu(m)

Move model or data `m` to the cpu. Makes
copies only if needed.
"""
cpu(m) = fmap(x -> adapt(Array, x), m)

"""
  gpu(m)

Move model or data `m` to the gpu device if available,
otherwise do nothing. Makes copies only if needed.
"""
gpu(m) = use_cuda[] ? fmap(CuArrays.cu, m) : m

# Precision

adapt_storage(T::Type{<:Real}, xs::AbstractArray{<:Real}) = convert.(T, xs)

paramtype(T::Type{<:Real}, m) = fmap(x -> adapt(T, x), m)

f32(m) = paramtype(Float32, m)
f64(m) = paramtype(Float64, m)
