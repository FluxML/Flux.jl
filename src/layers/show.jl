
for T in [
    :Chain, :Parallel, :SkipConnection,
    :Conv, :ConvTranspose, :CrossCor, :DepthwiseConv, :Dense,
    :BatchNorm, :LayerNorm, :InstanceNorm, :GroupNorm,
  ]
  @eval Base.show(io::IO, m::MIME"text/plain", x::$T) = _big_show(io, x)
end

function _big_show(io::IO, obj, indent=0, toclose=0)
  children = Flux.trainable(obj)
  if all(c -> isleaf(c) || isa(c,Tuple), children)  # need isa(c,Tuple) to get Conv right
    return _layer_show(io, obj, indent, toclose)
  end
  println(io, " "^indent, nameof(typeof(obj)), "(")
  for (i,c) in enumerate(children)
    close = i==length(children) && indent>0
    _big_show(io, c, indent+2, close ? toclose+1 : 0)
  end
  if indent == 0
    print(io, ")")
    _big_finale(io, params(obj))
  end
end

# Opt out of being printed as a container:
_big_show(io::IO, l::LayerNorm, i=0, t=0) = _layer_show(io, l, i, t)

# used both within Chain printing, and alone at top level.
function _layer_show(io::IO, layer, indent=0, toclose=0)
  str = sprint(show, layer, context=nothing) * ",)"^toclose
  print(io, " "^indent, str, indent==0 ? "" : ",")
  tab = indent==0 ? 20 : 39  # when inside Chain, move all parameter counts out to 40
  if !isempty(params(layer))
    print(" "^max(2, tab - indent - length(str)))
    pars = underscorise(sum(length, params(layer)))
    printstyled(io, "# ", pars, " parameters", color=:light_black)
    if !all(x -> all(isfinite, x), params(layer))
      printstyled(io, " (some NaN or Inf)", color=:red)
    elseif all(x -> all(iszero, x), params(layer))
      printstyled(io, " (all zero)", color=:cyan)
    end
  end
  indent==0 || println(io)
end

function _big_finale(io::IO, ps)
  num = length(ps)
  num < 3 && return println(io)
  pars = underscorise(sum(length, ps))
  bytes = sum(sizeof, ps)
  print(io, " "^19)
  printstyled(io, "# Total: ", num, " arrays, "; color=:light_black)
  printstyled(io, pars, " parameters, ", Base.format_bytes(bytes); color=:light_black)
end

underscorise(n::Integer) =
  join(reverse(join.(reverse.(Iterators.partition(digits(n), 3)))), '_')

# Zygote's containers

Base.show(io::IO, m::MIME"text/plain", p::Zygote.Params) = _param_show(io, p, true)

function _param_show(io::IO, p, iter::Bool)
  length(p) == 0 && return print(io, typeof(p), "([])")
  println(io, typeof(p), "([")
  ipad = length(string(length(p))) + 2
  spad = min(40-6-ipad, maximum(length∘summary, p))
  wid = get(io, :displaysize, (0,100))[2]
    for (i,x) in enumerate(p)
    if iter
        printstyled(io, "  ", lpad(string("[",i,"]"), ipad), color=:light_black)
    end
    desc = Base._truncate_at_width_or_chars(summary(x), spad)
    data = sprint(show, x, context=IOContext(io, :compact => true, :limit => true, :typeinfo => eltype(x)), sizehint=0)
    str = Base._truncate_at_width_or_chars(data, min(30, wid-40-12))
    print(io, "  ", rpad(desc, spad), "  ", str)
    if any(isnan, x)
      printstyled(io, "  (some NaN)", color=:red)
    elseif any(isinf, x)
      printstyled(io, "  (some Inf)", color=:red)
    elseif !isempty(x) && all(iszero, x)
      printstyled(io, "  (all zero)", color=:cyan)
    end
    println(io)
  end
  print(io, "])")
  pars = underscorise(sum(length, p))
  bytes = Base.format_bytes(sum(sizeof, p))
  printstyled(io, " "^18, "# Total: ", pars, " parameters, ", bytes; color=:light_black)
end

function Base.show(io::IO, m::MIME"text/plain", g::Zygote.Grads)
  println(io, "Zygote.Grads(")
  pars, bytes, spad = 0, 0, 0
  for k in keys(g.grads)
    pars += length(g[k])
    bytes += sizeof(g[k])
    spad = max(spad, length(summary(g[k])))
  end
  for k in keys(g.grads)
    x = g[k]
    str = sprint(show, x)
    str = length(str) < 32 ? str : str[1:32] * "…"
    print(io, "  ", rpad(summary(x), spad), "  ", str)
    if any(isnan, x)
      printstyled(io, "  (some NaN)", color=:red)
    elseif any(isinf, x)
      printstyled(io, "  (some Inf)", color=:red)
    elseif !isempty(x) && all(iszero, x)
      printstyled(io, "  (all zero)", color=:cyan)
    end
    println(io)
  end
  print(io, ")")
  printstyled(io, " "^19, "# Total: ", pars, " parameters, ", Base.format_bytes(bytes); color=:light_black)
end

