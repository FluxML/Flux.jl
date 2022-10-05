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

julia> outputsize(Dense(10, 4), (10,); padbatch=true)
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

julia> outputsize([Dense(10, 4), Dense(4, 2)], (10, 1)) # Vector of layers becomes a Chain
(2, 1)
```
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

julia> par = Parallel(vcat, Dense(5, 9), Dense(7, 11));

julia> Flux.outputsize(par, (5, 64), (7, 64))
(20, 64)

julia> m = Chain(par, Dense(20, 13), softmax);

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

for layer in (:LayerNorm, :BatchNorm, :InstanceNorm, :GroupNorm)
  @eval (l::$layer)(x::AbstractArray{Nil}) = x
end

## fixes for layers that don't work out of the box

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


export @autosize

"""
    @autosize (size...,) Chain(Layer(_ => 2), Layer(_), ...)

Returns the specified model, with each `_` replaced by an inferred number,
for input of the given size.

The unknown sizes are always the second-last dimension (or the length of a vector),
of that layer's input, which Flux usually regards as the channel dimension.
The underscore may appear as an argument of a layer, or inside a `=>`.

# Examples
```
julia> @autosize (3,) Chain(Dense(_ => 2, sigmoid), Flux.Scale(_))
Chain(
  Dense(3 => 2, σ),                     # 8 parameters
  Scale(2),                             # 4 parameters
)                   # Total: 4 arrays, 12 parameters, 304 bytes.

julia> img = [28, 28];

julia> @autosize (img..., 1, 32) Chain(              # size is only needed at runtime
          Chain(c = Conv((3,3), _ => 5; stride=2, pad=SamePad()),
                p = MeanPool((3,3)),
                b = BatchNorm(_),
                f = Flux.flatten),
          Dense(_ => _÷4, relu, init=Flux.rand32),   # can calculate output size _÷4
          SkipConnection(Dense(_ => _, relu), +),
          Dense(_ => 10),
       ) |> gpu                                      # moves to GPU after initialisation
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
* Won't work yet for Bilinear, except like `@autosize (5, 32) Flux.Bilinear(_ => 7)`
* Beyond a matrix it gets Dense wrong, e.g. `@autosize (2, 3, 4) Dense(_ => 5)`
* `LayerNorm(_,_)` probably won't work, needs first few input dims.
* RNN: `@autosize (7, 11) LSTM(_ => 5)` fails, but `outputsize(RNN(3=>7), (3,))` also fails.
"""
macro autosize(size, model)
  Meta.isexpr(size, :tuple) || error("@autosize's first argument must be a tuple, the size of the input")
  Meta.isexpr(model, :call) || error("@autosize's second argument must be something like Chain(layers...)")
  ex = makelazy(model)
  @gensym m
  quote
    $m = $ex
    $outputsize($m, $size)
    $striplazy($m)
  end |> esc
end

function makelazy(ex::Expr)
  n = underscoredepth(ex)
  n == 0 && return ex
  n == 1 && error("@autosize doesn't expect an underscore here: $ex")
  n == 2 && return :($LazyLayer($(string(ex)), $(makefun(ex)), nothing))
  n > 2 && return Expr(ex.head, ex.args[1], map(makelazy, ex.args[2:end])...)
end
makelazy(x) = x

function underscoredepth(ex::Expr)
  # Meta.isexpr(ex, :tuple) && :_ in ex.args && return 10
  ex.head in (:call, :kw, :(->), :block) || return 0
  ex.args[1] == :(=>) && ex.args[2] == :_ && return 1
  m = maximum(underscoredepth, ex.args)
  m == 0 ? 0 : m+1
end
underscoredepth(ex) = Int(ex == :_)

#=

@autosize (3,) Chain(one = Dense(_ => 10))  # needs kw
@autosize (10,) Maxout(() -> Dense(_ => 7, tanh), 3)  # needs ->, block

=#

function makefun(ex)
  @gensym s
  Expr(:(->), s, replaceunderscore(ex, s))
end

replaceunderscore(e, s) = e == :_ ? s : e
replaceunderscore(ex::Expr, s) = Expr(ex.head, map(a -> replaceunderscore(a, s), ex.args)...)

mutable struct LazyLayer
  str::String
  make::Function
  layer
end

function (l::LazyLayer)(x::AbstractArray)
  if l.layer != nothing
    return l.layer(x)
  end
  # s = channelsize(x)
  s = size(x, max(1, ndims(x)-1))
  lay = l.make(s)
  y = try
    lay(x)
  catch e
    @error l.str
    return nothing
  end
  l.layer = striplazy(lay)  # is this a good idea?
  return y
end

#=

Flux.outputsize(Chain(Dense(2=>3)), (4,))  # nice error
Flux.outputsize(Dense(2=>3), (4,))  # no nice error
@autosize (4,) Dense(2=>3)  # no nice error

@autosize (3,) Dense(2 => _)  # shouldn't work, weird error


@autosize (3,5,6) LayerNorm(_,_)  # no complaint, but
ans(rand(3,5,6))  # this fails



```
julia> Flux.outputsize(LayerNorm(2), (3,))
(3,)

julia> LayerNorm(2)(rand(Float32, 3))
ERROR: DimensionMismatch: arrays could not be broadcast to a common size; got a dimension with lengths 2 and 3

julia> BatchNorm(2)(fill(Flux.nil, 3)) |> size
(3,)

julia> BatchNorm(2)(rand(3))
ERROR: arraysize: dimension out of range
```


=#

# channelsize(x) = size(x, max(1, ndims(x)-1))

using Functors: functor, @functor

@functor LazyLayer # (layer,)

function striplazy(x)
  fs, re = functor(x)
  re(map(striplazy, fs))
end
striplazy(l::LazyLayer) = l.layer == nothing ? error("should be initialised!") : l.layer

# Could make LazyLayer usable outside of @autosize
# For instance allow @lazy

function Base.show(io::IO, l::LazyLayer)
  printstyled(io, "LazyLayer(", color=:light_black)
  if l.layer == nothing
    printstyled(io, l.str, color=:red)
  else
    printstyled(io, l.layer, color=:green)
  end
  printstyled(io, ")", color=:light_black)
end

_big_show(io::IO, l::LazyLayer, indent::Int=0, name=nothing) = _layer_show(io, l, indent, name)
