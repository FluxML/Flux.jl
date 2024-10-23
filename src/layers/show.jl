@nospecialize  # just for this file, for startup time

# This is called by @layer :expand, on layers which should be treated like Chain, and returns an expression:
function _macro_big_show(ex)
  quote
    # Entry point:
    function Base.show(io::IO, m::MIME"text/plain", x::$ex)
      if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        _big_show(io, x)
      elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
        _layer_show(io, x)
      else
        show(io, x)
      end
    end

    # Don't show Chain(Tuple(...)), always splat that. And ignore Recur's non-trainable state:
    Flux._show_children(x::$ex) = _flat_children(trainable(x))
  end
end

function _big_show(io::IO, obj, indent::Int=0, name=nothing)
  pre, post = _show_pre_post(obj)
  children = _show_children(obj)
  if all(_show_leaflike, children)
    # This check may not be useful anymore: it tries to infer when to stop the recursion by looking for grandkids,
    # but once all layers use @layer, they stop the recursion by defining a method for _big_show.
    _layer_show(io, obj, indent, name)
  else
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", pre)
    if obj isa Chain{<:NamedTuple} || obj isa NamedTuple
      # then we insert names -- can this be done more generically? 
      for k in Base.keys(obj)
        _big_show(io, obj[k], indent+2, k)
      end
    elseif obj isa Parallel{<:Any, <:NamedTuple} || obj isa PairwiseFusion{<:Any, <:NamedTuple}
      _big_show(io, obj.connection, indent+2)
      for k in Base.keys(obj)
        _big_show(io, obj[k], indent+2, k)
      end
    else
      for c in children
        _big_show(io, c, indent+2)
      end
    end
    if indent == 0  # i.e. this is the outermost container
      print(io, rpad(post, 2))
      _big_finale(io, obj)
    else
      println(io, " "^indent, post, ",")
    end
  end
end

_show_pre_post(obj) = string(nameof(typeof(obj)), "("), ")"
# _show_pre_post(::Chain{<:AbstractVector}) = "Chain([", "])"  # has to be done elsewhere
_show_pre_post(::AbstractVector) = "[", "]"
_show_pre_post(::NamedTuple) = "(;", ")"

_show_leaflike(x) = isleaf(x)  # mostly follow Functors, except for:

# note the covariance of tuple, using <:T causes warning or error
_show_leaflike(::Tuple{Vararg{Number}}) = true         # e.g. stride of Conv
_show_leaflike(::Tuple{Vararg{AbstractArray}}) = true  # e.g. parameters of LSTMcell
_show_leaflike(::AbstractArray{<:Number}) = true         # e.g. transposed arrays

_show_children(x) = trainable(x)
# This used to have methods for Chain, Maxout, Parallel, PairwiseFusion. Now @layer instead
# writes a method to use this function. It flattens the Tuple within Chain etc.
# (The remaining special cases are for printing of layer names when a NamedTuple, above.)
function _flat_children(x)
    alpha = map(f -> getfield(x, f), fieldnames(typeof(x)))
    beta = map(y -> y isa Union{Tuple, NamedTuple} ? y : (y,), alpha)
    gamma = ((beta...)...,)
end

# This is called by @layer, on layers which should be treated like Dense, and returns an expression:
function _macro_layer_show(ex)
  quote
    # Entry point:
    function Base.show(io::IO, m::MIME"text/plain", x::$ex)
      if !get(io, :compact, false)
        _layer_show(io, x)
      else
        show(io, x)
      end
    end

    # Exit from _big_show recursion:
    Flux._big_show(io::IO, obj::$ex, indent::Int=0, name=nothing) = _layer_show(io, obj, indent, name)
  end
end

function _layer_show(io::IO, layer, indent::Int=0, name=nothing)
  _str = isnothing(name) ? "" : "$name = "
  str = _str * _layer_string(io, layer)
  print(io, " "^indent, str, indent==0 ? "" : ",")
  if !isempty(params(layer))
    print(io, " "^max(2, (indent==0 ? 20 : 39) - indent - length(str)))
    printstyled(io, "# ", underscorise(sum(length, params(layer); init=0)), " parameters"; 
color=:light_black)
    nonparam = _childarray_sum(length, layer) - sum(length, params(layer), init=0)
    if nonparam > 0
      printstyled(io, ", plus ", underscorise(nonparam), indent==0 ? " non-trainable" : ""; color=:light_black)
    end
    _nan_show(io, params(layer))
  end
  indent==0 || println(io)
end

_layer_string(io::IO, layer) = sprint(show, layer, context=io)
# _layer_string(::IO, a::AbstractArray) = summary(layer)  # sometimes too long e.g. CuArray
function _layer_string(::IO, a::AbstractArray)
  full = string(typeof(a))
  comma = findfirst(',', full)
  short = isnothing(comma) ? full : full[1:comma] * "...}"
  Base.dims2string(size(a)) * " " * short
end

function _big_finale(io::IO, m)
  ps = params(m)
  if length(ps) > 2
    pars = underscorise(sum(length, ps; init=0))
    bytes = Base.format_bytes(Base.summarysize(m))
    noncnt = _childarray_sum(_->1, m) - length(ps)
    if noncnt > 0
      nonparam = underscorise(_childarray_sum(length, m) - sum(length, ps; init=0))
      printstyled(io, " "^08, "# Total: ", length(ps), " trainable arrays, "; color=:light_black)
      println(io, pars, " parameters,")
      printstyled(io, " "^10, "# plus ", noncnt, " non-trainable, ", nonparam, " parameters, summarysize "; color=:light_black)
      print(io, bytes, ".")
    else
      printstyled(io, " "^18, "# Total: ", length(ps), " arrays, "; color=:light_black)
      print(io, pars, " parameters, ", bytes, ".")
    end
  end
end

_childarray_sum(f, x::AbstractArray{<:Number}) = f(x)
_childarray_sum(f, x) = isleaf(x) ? 0 : sum(y -> _childarray_sum(f, y), Functors.children(x), 
init=0)

# utility functions

underscorise(n::Integer) =
  join(reverse(join.(reverse.(Iterators.partition(digits(n), 3)))), '_')

function _nan_show(io::IO, x)
  if !isempty(x) && _all(iszero, x)
    printstyled(io, "  (all zero)", color=:cyan)
  elseif _any(isnan, x)
    printstyled(io, "  (some NaN)", color=:red)
  elseif _any(isinf, x)
    printstyled(io, "  (some Inf)", color=:red)
  end
end

@specialize  # un-does @nospecialze at the top of this file

_any(f, xs::AbstractArray{<:Number}) = any(f, xs)
# _any(f, xs::Union{Tuple,NamedTuple,Zygote.Params}) = any(x -> _any(f, x), xs)
_any(f, xs) = any(x -> _any(f, x), xs)
_any(f, x::Number) = f(x)
# _any(f, x) = false

_all(f, xs) = !_any(!f, xs)

#=

julia> struct Tmp2; x; y; end; Flux.@functor Tmp2

# Before, notice Array(), NamedTuple(), and values

julia> Chain(Tmp2([Dense(2,3), randn(3,4)'], (x=1:3, y=Dense(3,4), z=rand(3))))
Chain(
  Tmp2(
    Array(
      Dense(2 => 3),                    # 9 parameters
      [0.351978391016603 0.6408681372462821 -1.326533184688648; 0.09481930831795712 1.430103476272605 0.7250467613675332; 2.03372151428719 -0.015879812799495713 1.9499692162118236; -1.6346846180722918 -0.8364610153059454 -1.2907265737483433],  # 12 parameters
    ),
    NamedTuple(
      1:3,                              # 3 parameters
      Dense(3 => 4),                    # 16 parameters
      [0.9666158193429335, 0.01613900990539574, 0.0205920186127464],  # 3 parameters
    ),
  ),
)                   # Total: 7 arrays, 43 parameters, 644 bytes.

# After, (; x=, y=, z=) and "3-element Array"

julia> Chain(Tmp2([Dense(2,3), randn(3,4)'], (x=1:3, y=Dense(3,4), z=rand(3))))
Chain(
  Tmp2(
    [
      Dense(2 => 3),                    # 9 parameters
      4×3 Adjoint,                      # 12 parameters
    ],
    (;
      x = 3-element UnitRange,          # 3 parameters
      y = Dense(3 => 4),                # 16 parameters
      z = 3-element Array,              # 3 parameters
    ),
  ),
)                   # Total: 7 arrays, 43 parameters, 644 bytes.

=#
