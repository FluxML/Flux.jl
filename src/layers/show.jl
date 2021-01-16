
for T in [
    :Chain, :Parallel, :SkipConnection, :Recur,
    :Conv, :ConvTranspose, :CrossCor, :DepthwiseConv, :Dense,
    :BatchNorm, :LayerNorm, :InstanceNorm, :GroupNorm,
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

function _big_show(io::IO, obj, indent::Int=0)
  children = trainable(obj)
  if all(c -> isleaf(c) || _show_leaflike(c), children)
    return _layer_show(io, obj, indent)
  end
  println(io, " "^indent, nameof(typeof(obj)), "(")
  for (i,c) in enumerate(children)
    _big_show(io, c, indent+2)
  end
  if indent == 0
    print(io, ")")
    _big_finale(io, params(obj))
  else
    println(io, " "^indent, "),")
  end
end

_show_leaflike(::Any) = false
_show_leaflike(::Tuple{Vararg{<:Number}}) = true  # e.g. stride of Conv
_show_leaflike(::Tuple{Vararg{<:AbstractArray}}) = true  # e.g. parameters of LSTMcell
_show_leaflike(::Diagonal) = true  # appears inside LayerNorm, but let's collapse that.

function _layer_show(io::IO, layer, indent::Int=0)
  # str = sprint(show, layer, context=io)
  str = string(layer)
  print(io, " "^indent, str, indent==0 ? "" : ",")
  if !isempty(params(layer))
    print(io, " "^max(2, (indent==0 ? 20 : 39) - indent - length(str)))
    printstyled(io, "# ", underscorise(sum(length, params(layer))), " parameters"; color=:light_black)
    _nan_show(io, params(layer))
  end
  indent==0 || println(io)
end

function _big_finale(io::IO, ps)
  length(ps) < 3 && return
  pars = underscorise(sum(length, ps))
  bytes = Base.format_bytes(sum(sizeof, ps))
  printstyled(io, " "^19, "# Total: ", length(ps), " arrays, ", pars, " parameters, ", bytes; color=:light_black)
end

# Zygote's containers

function Base.show(io::IO, m::MIME"text/plain", p::Zygote.Params)
  get(io, :typeinfo, nothing) === nothing ? _param_show(io, p) : show(io, p)
end

function _param_show(io::IO, p)
  length(p) == 0 && return print(io, typeof(p), "([])")
  println(io, typeof(p), "([")
  ipad = length(string(length(p))) + 2
  spad = min(50-6-ipad, maximum(length∘summary, p))
  wid = get(io, :displaysize, (0,100))[2] # not certain this is working
  for (i,x) in enumerate(p)
    printstyled(io, "  ", lpad(string("[",i,"]"), ipad), color=:light_black)
    desc = Base._truncate_at_width_or_chars(summary(x), spad)
    data = sprint(show, x, context=IOContext(io, :compact => true, :limit => true, :typeinfo => eltype(x)), sizehint=0)
    str = Base._truncate_at_width_or_chars(data, min(30, wid-40-12))
    print(io, "  ", rpad(desc, spad), "  ", str)
    _nan_show(io, x)
    println(io)
  end
  print(io, "])")
  pars = underscorise(sum(length, p))
  bytes = Base.format_bytes(sum(sizeof, p))
  printstyled(io, " "^18, "# Total: ", pars, " parameters, ", bytes; color=:light_black)
end

function Base.show(io::IO, m::MIME"text/plain", g::Zygote.Grads)
  get(io, :typeinfo, nothing) === nothing ? _grad_show(io, g) : show(io, g)
end

function _grad_show(io::IO, g)
  println(io, "Zygote.Grads(")
  pars, bytes, spad = 0, 0, 0
  for k in keys(g.grads)
    pars += length(g[k])
    bytes += sizeof(g[k])
    spad = max(spad, length(summary(g[k])))
  end
  for k in keys(g.grads)
    x = g[k]
    str = Base._truncate_at_width_or_chars(sprint(show, x), 32) # ??
    # print(io, "  ", rpad(summary(x), spad), "  ", str)
    print(io, "  ", rpad(summary(x), 20-4), "  ", str)
    _nan_show(io, x)
    println(io)
  end
  print(io, ")")
  printstyled(io, " "^19, "# Total: ", pars, " parameters, ", Base.format_bytes(bytes); color=:light_black)
end

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
_any(f, xs::Union{Tuple,NamedTuple,Zygote.Params}) = any(x -> _any(f, x), xs)
_any(f, x::Number) = f(x)
_any(f, x) = false

_all(f, xs) = !_any(!f, xs)
