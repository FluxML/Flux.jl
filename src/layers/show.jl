
for T in [
    :Chain, :Parallel, :SkipConnection, :Recur, :Maxout, :PairwiseFusion  # container types
  ]
  @eval function Base.show(io::IO, m::MIME"text/plain", x::$T)
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
      _big_show(io, x)
    elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
      _layer_show(io, x)
    else
      show(io, x)
    end
  end
end

function _big_show(io::IO, obj, indent::Int=0, name=nothing)
  pre, post = obj isa Chain{<:AbstractVector} ? ("([", "])") : ("(", ")")
  children = _show_children(obj)
  if all(_show_leaflike, children)
    _layer_show(io, obj, indent, name)
  else
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", nameof(typeof(obj)), pre)
    if obj isa Chain{<:NamedTuple} && children == getfield(obj, :layers)
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

_show_leaflike(x) = isleaf(x)  # mostly follow Functors, except for:

# note the covariance of tuple, using <:T causes warning or error
_show_leaflike(::Tuple{Vararg{Number}}) = true         # e.g. stride of Conv
_show_leaflike(::Tuple{Vararg{AbstractArray}}) = true  # e.g. parameters of LSTMcell
_show_leaflike(::Scale) = true                           # appears inside LayerNorm
_show_leaflike(::AbstractArray{<:Number}) = true         # e.g. transposed arrays

_show_children(x) = trainable(x)  # except for layers which hide their Tuple:
_show_children(c::Chain) = c.layers
_show_children(m::Maxout) = m.layers
_show_children(p::Parallel) = (p.connection, p.layers...)
_show_children(f::PairwiseFusion) = (f.connection, f.layers...)

for T in [
    :Conv, :ConvTranspose, :CrossCor, :Dense, :Scale, :Bilinear, :Embedding, :EmbeddingBag,
    :BatchNorm, :LayerNorm, :InstanceNorm, :GroupNorm,
  ]
  @eval function Base.show(io::IO, m::MIME"text/plain", x::$T)
    if !get(io, :compact, false)
      _layer_show(io, x)
    else
      show(io, x)
    end
  end
end

function _layer_show(io::IO, layer, indent::Int=0, name=nothing)
  _str = isnothing(name) ? "" : "$name = "
  str = _str * sprint(show, layer, context=io)
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

_any(f, xs::AbstractArray{<:Number}) = any(f, xs)
# _any(f, xs::Union{Tuple,NamedTuple,Zygote.Params}) = any(x -> _any(f, x), xs)
_any(f, xs) = any(x -> _any(f, x), xs)
_any(f, x::Number) = f(x)
# _any(f, x) = false

_all(f, xs) = !_any(!f, xs)
