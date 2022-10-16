module NilNumber

using NNlib
import Random

"""
    nil = Nil()

`Nil` is a singleton type with a single instance `nil`.
Unlike `Nothing` and `Missing` it is a number: `Nil <: Real <: Number`.
"""
struct Nil <: Real end

@doc @doc(Nil)
const nil = Nil()

Nil(::T) where T<:Number = nil
(::Type{T})(::Nil) where T<:Number = nil
Base.convert(::Type{Nil}, ::Number) = nil

Base.float(::Type{Nil}) = Nil

for f in [:copy, :zero, :one, :oneunit,
          :+, :-, :abs, :abs2, :inv,
          :exp, :log, :log1p, :log2, :log10,
          :sqrt, :tanh, :conj]
  @eval Base.$f(::Nil) = nil
end

for f in [:+, :-, :*, :/, :^, :mod, :div, :rem]
  @eval Base.$f(::Nil, ::Nil) = nil
end

Base.:<(::Nil, ::Nil) = true
Base.:<=(::Nil, ::Nil) = true

Base.isnan(::Nil) = false
Base.isfinite(::Nil) = true
Base.typemin(::Type{Nil}) = nil
Base.typemax(::Type{Nil}) = nil

Base.promote_rule(x::Type{Nil}, y::Type{<:Number}) = Nil

Random.rand(rng::Random.AbstractRNG, ::Random.SamplerType{Nil}) = nil

end  # module

using .NilNumber: Nil, nil

"""
    outputsize(m, inputsize::Tuple; padbatch=false)

Calculate the size of the output from model `m`, given the size of the input.
Obeys `outputsize(m, size(x)) == size(m(x))` for valid input `x`.

Keyword `padbatch=true` is equivalent to using `(inputsize..., 1)`, and
returns the final size including this extra batch dimension.

This should be faster than calling `size(m(x))`. It uses a trivial number type,
which should work out of the box for custom layers.

If `m` is a `Tuple` or `Vector`, its elements are applied in sequence, like `Chain(m...)`.

# Examples
```julia-repl
julia> using Flux: outputsize

julia> outputsize(Dense(10 => 4), (10,); padbatch=true)
(4, 1)

julia> m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32));

julia> m(randn(Float32, 10, 10, 3, 64)) |> size
(6, 6, 32, 64)

julia> outputsize(m, (10, 10, 3); padbatch=true)
(6, 6, 32, 1)

julia> outputsize(m, (10, 10, 3, 64))
(6, 6, 32, 64)

julia> try outputsize(m, (10, 10, 7, 64)) catch e println(e) end
┌ Error: layer Conv((3, 3), 3=>16), index 1 in Chain, gave an error with input of size (10, 10, 7, 64)
└ @ Flux ~/.julia/dev/Flux/src/outputsize.jl:114
DimensionMismatch("Input channels must match! (7 vs. 3)")

julia> outputsize([Dense(10 => 4), Dense(4 => 2)], (10, 1)) # Vector of layers becomes a Chain
(2, 1)
```

Limitations:
* `Embedding` accepts either integers or one-hot arrays, and `ohx = onehotbatch(x, ...)`
  has one more dimension than `x`. Here `outputsize` uses `size(x)`.
* At present `outputsize` does not work with recurrent layers,
  `outputsize(RNN(2 => 3), (2, 1))` gives an error. This is a bug.
"""
function outputsize(m, inputsizes::Tuple...; padbatch=false)
  x = nil_input(padbatch, inputsizes...)
  return size(m(x))
end

nil_input(pad::Bool, s::Tuple{Vararg{Integer}}) = pad ? fill(nil, (s...,1)) : fill(nil, s)
nil_input(pad::Bool, multi::Tuple{Vararg{Integer}}...) = nil_input.(pad, multi)
nil_input(pad::Bool, tup::Tuple{Vararg{Tuple}}) = nil_input(pad, tup...)

function outputsize(m::Chain, inputsizes::Tuple{Vararg{Integer}}...; padbatch=false)
  x = nil_input(padbatch, inputsizes...)
  for (i,lay) in enumerate(m.layers)
    try
      x = lay(x)
    catch err
      str = x isa AbstractArray ? "with input of size $(size(x))" : ""
      @error "layer $lay, index $i in Chain, gave an error $str"
      rethrow(err)
    end
  end
  return size(x)
end

"""
    outputsize(m, x_size, y_size, ...; padbatch=false)

For model or layer `m` accepting multiple arrays as input,
this returns `size(m((x, y, ...)))` given `size_x = size(x)`, etc.

# Examples
```jldoctest
julia> x, y = rand(Float32, 5, 64), rand(Float32, 7, 64);

julia> par = Parallel(vcat, Dense(5 => 9), Dense(7 => 11));

julia> Flux.outputsize(par, (5, 64), (7, 64))
(20, 64)

julia> m = Chain(par, Dense(20 => 13), softmax);

julia> Flux.outputsize(m, (5,), (7,); padbatch=true)
(13, 1)

julia> par(x, y) == par((x, y)) == Chain(par, identity)((x, y))
true
```
Notice that `Chain` only accepts multiple arrays as a tuple,
while `Parallel` also accepts them as multiple arguments;
`outputsize` always supplies the tuple.
"""
outputsize

## make tuples and vectors be like Chains

outputsize(m::Tuple, input::Tuple...; padbatch=false) = outputsize(Chain(m...), input...; padbatch=padbatch)
outputsize(m::AbstractVector, input::Tuple...; padbatch=false) = outputsize(Chain(m...), input...; padbatch=padbatch)

## bypass statistics in normalization layers

for layer in (:BatchNorm, :InstanceNorm, :GroupNorm)  # LayerNorm works fine
  @eval function (l::$layer)(x::AbstractArray{Nil})
    l.chs == size(x, ndims(x)-1) || throw(DimensionMismatch(
      string($layer, " expected ", l.chs, " channels, but got size(x) == ", size(x))))
    x
  end
end

## fixes for layers that don't work out of the box

(m::Embedding)(x::Nil) = similar(m.weight, Nil, size(m.weight, 1))
(m::Embedding)(x::AbstractArray{Nil}) = similar(m.weight, Nil, size(m.weight, 1), size(x)...)

for (fn, Dims) in ((:conv, DenseConvDims),)
  @eval begin
    function NNlib.$fn(a::AbstractArray{Nil}, b::AbstractArray{Nil}, dims::$Dims)
      fill(nil, NNlib.output_size(dims)..., NNlib.channels_out(dims), size(a)[end])
    end

    function NNlib.$fn(a::AbstractArray{<:Real}, b::AbstractArray{Nil}, dims::$Dims)
      NNlib.$fn(fill(nil, size(a)), b, dims)
    end

    function NNlib.$fn(a::AbstractArray{Nil}, b::AbstractArray{<:Real}, dims::$Dims)
      NNlib.$fn(a, fill(nil, size(b)), dims)
    end
  end
end


"""
    @autosize (size...,) Chain(Layer(_ => 2), Layer(_), ...)

Returns the specified model, with each `_` replaced by an inferred number,
for input of the given `size`.

The unknown sizes are usually the second-last dimension of that layer's input,
which Flux regards as the channel dimension.
(A few layers, `Dense` & [`LayerNorm`](@ref), instead always use the first dimension.)
The underscore may appear as an argument of a layer, or inside a `=>`.
It may be used in further calculations, such as `Dense(_ => _÷4)`.

# Examples
```
julia> @autosize (3, 1) Chain(Dense(_ => 2, sigmoid), BatchNorm(_, affine=false))
Chain(
  Dense(3 => 2, σ),                     # 8 parameters
  BatchNorm(2, affine=false),
) 

julia> img = [28, 28];

julia> @autosize (img..., 1, 32) Chain(              # size is only needed at runtime
          Chain(c = Conv((3,3), _ => 5; stride=2, pad=SamePad()),
                p = MeanPool((3,3)),
                b = BatchNorm(_),
                f = Flux.flatten),
          Dense(_ => _÷4, relu, init=Flux.rand32),   # can calculate output size _÷4
          SkipConnection(Dense(_ => _, relu), +),
          Dense(_ => 10),
       )
Chain(
  Chain(
    c = Conv((3, 3), 1 => 5, pad=1, stride=2),  # 50 parameters
    p = MeanPool((3, 3)),
    b = BatchNorm(5),                   # 10 parameters, plus 10
    f = Flux.flatten,
  ),
  Dense(80 => 20, relu),                # 1_620 parameters
  SkipConnection(
    Dense(20 => 20, relu),              # 420 parameters
    +,
  ),
  Dense(20 => 10),                      # 210 parameters
)         # Total: 10 trainable arrays, 2_310 parameters,
          # plus 2 non-trainable, 10 parameters, summarysize 10.469 KiB.

julia> outputsize(ans, (28, 28, 1, 32))
(10, 32)
```

Limitations:
* While `@autosize (5, 32) Flux.Bilinear(_ => 7)` is OK, something like `Bilinear((_, _) => 7)` will fail.
* While `Scale(_)` and `LayerNorm(_)` are fine (and use the first dimension), `Scale(_,_)` and `LayerNorm(_,_)`
  will fail if `size(x,1) != size(x,2)`.
* RNNs won't work: `@autosize (7, 11) LSTM(_ => 5)` fails, because `outputsize(RNN(3=>7), (3,))` also fails, a known issue.
"""
macro autosize(size, model)
  Meta.isexpr(size, :tuple) || error("@autosize's first argument must be a tuple, the size of the input")
  Meta.isexpr(model, :call) || error("@autosize's second argument must be something like Chain(layers...)")
  ex = _makelazy(model)
  @gensym m
  quote
    $m = $ex
    $outputsize($m, $size)
    $striplazy($m)
  end |> esc
end

function _makelazy(ex::Expr)
  n = _underscoredepth(ex)
  n == 0 && return ex
  n == 1 && error("@autosize doesn't expect an underscore here: $ex")
  n == 2 && return :($LazyLayer($(string(ex)), $(_makefun(ex)), nothing))
  n > 2 && return Expr(ex.head, map(_makelazy, ex.args)...)
end
_makelazy(x) = x

function _underscoredepth(ex::Expr)
  # Meta.isexpr(ex, :tuple) && :_ in ex.args && return 10
  ex.head in (:call, :kw, :(->), :block, :parameters)  || return 0
  ex.args[1] === :(=>) && ex.args[2] === :_ && return 1
  m = maximum(_underscoredepth, ex.args)
  m == 0 ? 0 : m+1
end
_underscoredepth(ex) = Int(ex === :_)

function _makefun(ex)
  T = Meta.isexpr(ex, :call) ? ex.args[1] : Type
  @gensym x s
  Expr(:(->), x, Expr(:block, :($s = $autosizefor($T, $x)), _replaceunderscore(ex, s)))
end

"""
    autosizefor(::Type, x)

If an `_` in your layer's constructor, used within `@autosize`, should
*not* mean the 2nd-last dimension, then you can overload this.

For instance `autosizefor(::Type{<:Dense}, x::AbstractArray) = size(x, 1)`
is needed to make `@autosize (2,3,4) Dense(_ => 5)` return 
`Dense(2 => 5)` rather than `Dense(3 => 5)`.
"""
autosizefor(::Type, x::AbstractArray) = size(x, max(1, ndims(x)-1))
autosizefor(::Type{<:Dense}, x::AbstractArray) = size(x, 1)
autosizefor(::Type{<:LayerNorm}, x::AbstractArray) = size(x, 1)

autosizefor(::Type{<:Embedding}, x::AbstractArray) = error(
  "@autosize Embeeding(_ => n) cannot work, as this _ is the size of the vocabulary, not an array size")

_replaceunderscore(e, s) = e === :_ ? s : e
_replaceunderscore(ex::Expr, s) = Expr(ex.head, map(a -> _replaceunderscore(a, s), ex.args)...)

mutable struct LazyLayer
  str::String
  make::Function
  layer
end

function (l::LazyLayer)(x::AbstractArray, ys::AbstractArray...)
  l.layer === nothing || return l.layer(x, ys...)
  made = l.make(x)  # for something like `Bilinear((_,__) => 7)`, perhaps need `make(xy...)`, later.
  y = made(x, ys...)
  l.layer = made  # mutate after we know that call worked
  return y
end

function striplazy(m)
  fs, re = functor(m)
  re(map(striplazy, fs))
end
function striplazy(l::LazyLayer)
  l.layer === nothing || return l.layer
  error("LazyLayer should be initialised, e.g. by outputsize(model, size), before using stiplazy")
end

# Could make LazyLayer usable outside of @autosize, for instance allow Chain(@lazy Dense(_ => 2))?
# But then it will survive to produce weird structural gradients etc. 

function ChainRulesCore.rrule(l::LazyLayer, x)
  l(x), _ -> error("LazyLayer should never be used within a gradient. Call striplazy(model) first to remove all.")
end
function ChainRulesCore.rrule(::typeof(striplazy), m)
  striplazy(m), _ -> error("striplazy should never be used within a gradient")
end

params!(p::Params, x::LazyLayer, seen = IdSet()) = error("LazyLayer should never be used within params(m). Call striplazy(m) first.")

Functors.functor(::Type{<:LazyLayer}, x) = error("LazyLayer should not be walked with Functors.jl, as the arrays which Flux.gpu wants to move may not exist yet.")

function Base.show(io::IO, l::LazyLayer)
  printstyled(io, "LazyLayer(", color=:light_black)
  if l.layer == nothing
    printstyled(io, l.str, color=:magenta)
  else
    printstyled(io, l.layer, color=:cyan)
  end
  printstyled(io, ")", color=:light_black)
end

_big_show(io::IO, l::LazyLayer, indent::Int=0, name=nothing) = _layer_show(io, l, indent, name)
