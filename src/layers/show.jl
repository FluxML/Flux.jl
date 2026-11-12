# Pretty-printing of models: the tree of layers, with parameter counts in gray.
#
# None of this should need compiling anew for each model type, or else the first display of
# every new model (and of every model moved to the GPU) would take a second. So:
# * Methods here are `@nospecializeinfer`, inferred & compiled once for `Any`, not per type.
# * Children are read with `nfields` & `getfield`, rather than `map`, `all`, `sum` or closures.
# * Structs using the default `Functors.functor` and `Optimisers.trainable`, as `@layer` does,
#   have their fields read directly, see `_show_functor_children`.
# * What's left is per-type work in small methods reached by dynamic dispatch, mostly the
#   2-arg `show` of each layer, and any custom `trainable` or `functor` methods.

@nospecialize  # for every method in this file, until `@specialize` below

# The methods written by @layer only call the functions below, so need not be compiled for
# each concrete type. For this they want `@nospecialize(x::T)` in their signatures, which is
# simplest to write in a fully escaped block, as `esc` within `Expr(:meta, ...)` is not allowed.
_unesc(ex::Expr) = ex.head === :escape ? _unesc(ex.args[1]) : Expr(ex.head, map(_unesc, ex.args)...)
_unesc(ex) = ex
_nospec(x::Symbol, type) = Expr(:meta, :nospecialize, :($x::$(_unesc(type))))

# This is called by @layer and returns an expression:
function _macro_big_show(type)
  esc(quote
    # Entry point:
    $Base.show(io::$IO, ::$(MIME"text/plain"), $(_nospec(:x, type))) = $_show_expanded(io, x)

    # Don't show Chain(Tuple(...)), always splat that. And ignore non-trainable buffers:
    $Flux._show_children($(_nospec(:x, type))) = $_flat_children($_show_trainable(x))
  end)
end

# This is called by @layer :named and returns an expression:
function _macro_named_show(type)
  esc(quote
    $Base.show(io::$IO, ::$(MIME"text/plain"), $(_nospec(:x, type))) = $_show_expanded(io, x)

    $Flux._show_children($(_nospec(:x, type))) = $_show_functor_children(x)
  end)
end

# This is called by @layer :noexpand, on layers which should be treated like Dense, and returns an expression:
function _macro_layer_show(type)
  esc(quote
    # Entry point:
    $Base.show(io::$IO, ::$(MIME"text/plain"), $(_nospec(:x, type))) = $_show_unexpanded(io, x)

    # Exit from _big_show recursion:
    $Flux._big_show(io::$IO, $(_nospec(:x, type)), indent::$Int=0, name=$nothing) = $_layer_show(io, x, indent, name)
  end)
end

Base.@nospecializeinfer function _show_expanded(io::IO, x)
  if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
    _buffered(_big_show, io, x)
  elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
    _buffered(_layer_show, io, x)
  else
    show(io, x)
  end
end

Base.@nospecializeinfer function _show_unexpanded(io::IO, x)
  if !get(io, :compact, false)
    _buffered(_layer_show, io, x)
  else
    show(io, x)
  end
end

# Printing into an IOBuffer, then copying to `io`, means that all the printing code is compiled for
# just one type of IO, which can then be precompiled. It keeps the properties of `io`, e.g. :color.
# What's printed is copied to `io` after each child of the outermost container (see `_big_show_child`),
# and when some layer's `show` throws, so that the output still arrives a piece at a time.
Base.@nospecializeinfer function _buffered(f, io::IO, x)
  buf = IOBuffer()
  try
    f(IOContext(IOContext(buf, io), :_flux_show_sink => io), x)
  finally
    write(io, take!(buf))
  end
  return nothing
end

Base.@nospecializeinfer function _flush_buffered(io::IO)
  sink = get(io, :_flux_show_sink, nothing)
  if sink isa IO && io isa IOContext && io.io isa IOBuffer
    write(sink, take!(io.io))
  end
  return nothing
end

Base.@nospecializeinfer function _big_show_child(io::IO, c, indent::Int, name=nothing)
  _big_show(io, c, indent+2, name)
  indent == 0 && _flush_buffered(io)
  return nothing
end

Base.@nospecializeinfer function _big_show(io::IO, obj, indent::Int=0, name=nothing)
  children = _show_children(obj)
  # A Dict's pairs are printed, as before, although `_childvec` (for counting) takes its values:
  kids = children isa AbstractDict ? collect(Any, children) : _childvec(children)
  if _all_leaflike(kids)
    # This check may not be useful anymore: it tries to infer when to stop the recursion by looking for grandkids,
    # but once all layers use @layer, they stop the recursion by defining a method for _big_show.
    _layer_show(io, obj, indent, name)
  else
    pre, post = _show_pre_post(obj)
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", pre)
    indent == 0 && _flush_buffered(io)
    if obj isa Chain{<:NamedTuple} || obj isa NamedTuple
      # then we insert names -- can this be done more generically?
      layers = obj isa NamedTuple ? obj : obj.layers
      for k in keys(layers)::Tuple
        _big_show_child(io, getfield(layers, k), indent, k)
      end
    elseif obj isa Parallel{<:Any, <:NamedTuple} || obj isa PairwiseFusion{<:Any, <:NamedTuple}
      _big_show_child(io, obj.connection, indent)
      for k in keys(obj.layers)::Tuple
        _big_show_child(io, getfield(obj.layers, k), indent, k)
      end
    elseif children isa NamedTuple
      for k in keys(children)::Tuple
        _big_show_child(io, getfield(children, k), indent, k)
      end
    elseif children isa _ShowFields
      for i in eachindex(children.vals)
        _big_show_child(io, children.vals[i], indent, children.names[i])
      end
    else
      for c in kids
        _big_show_child(io, c, indent)
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

for Fix in (:Fix1, :Fix2)
  pre = string(Fix, "(")
  @eval Base.@nospecializeinfer function _big_show(io::IO, obj::Base.$Fix, indent::Int=0, name=nothing)
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", $pre)
    _big_show(io, obj.f, indent+2)
    _big_show(io, obj.x, indent+2)
    println(io, " "^indent, ")", ",")
  end
end

_show_pre_post(obj) = string(nameof(typeof(obj)), "("), ")"
_show_pre_post(::AbstractVector) = "[", "]"
_show_pre_post(::NamedTuple) = "(;", ")"

_show_leaflike(x) = _show_isleaf(x)  # mostly follow Functors, except for:

# note the covariance of tuple, using <:T causes warning or error
_show_leaflike(::Tuple{Vararg{Number}}) = true         # e.g. stride of Conv
_show_leaflike(::Tuple{Vararg{AbstractArray}}) = true  # e.g. parameters of LSTMcell
_show_leaflike(::Tuple{}) = true                       # e.g. empty tuple
_show_leaflike(::AbstractArray{<:Number}) = true       # e.g. transposed arrays

Base.@nospecializeinfer function _all_leaflike(kids::Vector{Any})
  for c in kids
    _show_leaflike(c)::Bool || return false
  end
  return true
end

_show_children(x) = _show_trainable(x)
# This used to have methods for Chain, Maxout, Parallel, PairwiseFusion. Now @layer instead
# writes a method to use this function. It flattens the Tuple within Chain etc.
# (The remaining special cases are for printing of layer names when a NamedTuple, above.)
Base.@nospecializeinfer function _flat_children(x)
  out = Any[]
  for y in _childvec(x)
    if y isa Union{Tuple, NamedTuple}
      for j in 1:nfields(y)
        push!(out, getfield(y, j))
      end
    else
      push!(out, y)
    end
  end
  return out
end

# The children of a node, as returned by `trainable` or `Functors.children`, in a Vector{Any}:
# Nothing mutates the result, so it may be `ch` itself.
Base.@nospecializeinfer function _childvec(ch)
  if ch isa _ShowFields
    return ch.vals
  elseif ch isa Vector{Any}
    return ch
  elseif ch isa Union{Tuple, NamedTuple}
    out = Vector{Any}(undef, nfields(ch))
    for i in eachindex(out)
      out[i] = getfield(ch, i)
    end
    return out
  elseif ch isa AbstractDict
    return collect(Any, values(ch))
  else
    return collect(Any, ch)
  end
end

Base.@nospecializeinfer function _layer_show(io::IO, layer, indent::Int=0, name=nothing)
  _str = isnothing(name) ? "" : "$name = "
  str = _str * _layer_string(io, layer)
  print(io, " "^indent, str, indent==0 ? "" : ",")
  ps = _show_trainables(layer)
  if !isempty(ps)
    npars = _sum_length(ps)
    print(io, " "^max(2, (indent==0 ? 20 : 39) - indent - length(str)))
    printstyled(io, "# ", underscorise(npars), " parameters"; color=:light_black)
    nonparam = _show_arrays(layer)[2] - npars
    if nonparam > 0
      printstyled(io, ", plus ", underscorise(nonparam), indent==0 ? " non-trainable" : ""; color=:light_black)
    end
    _nan_show(io, ps)
  end
  indent==0 || println(io)
end

Base.@nospecializeinfer function _layer_string(io::IO, layer)
  buf = IOBuffer()
  show(IOContext(buf, io), layer)
  return String(take!(buf))
end
# _layer_string(::IO, a::AbstractArray) = summary(layer)  # sometimes too long e.g. CuArray
Base.@nospecializeinfer function _layer_string(::IO, a::AbstractArray)
  full = string(typeof(a))
  comma = findfirst(',', full)
  short = isnothing(comma) ? full : full[1:comma] * "...}"
  Base.dims2string(size(a)) * " " * short
end

Base.@nospecializeinfer function _big_finale(io::IO, m)
  ps = _show_trainables(m)
  if length(ps) > 2
    npars = _sum_length(ps)
    pars = underscorise(npars)
    bytes = Base.format_bytes(Base.summarysize(m))
    narrays, nlength = _show_arrays(m)
    noncnt = narrays - length(ps)
    if noncnt > 0
      nonparam = underscorise(nlength - npars)
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

# The same arrays as `Optimisers.trainables(x)`, without compiling an `fmap` for each type of `x`:
# numeric arrays reached through `trainable`, with tied weights appearing once.
Base.@nospecializeinfer function _show_trainables(x)
  ps = Any[]
  _show_trainables!(ps, Base.IdSet{Any}(), x)
  return ps
end

Base.@nospecializeinfer function _show_trainables!(ps::Vector{Any}, seen::Base.IdSet{Any}, x)
  m = _show_methods(typeof(x))
  if _show_isnumeric(x, m)
    if _anymutable(x)  # like the cache of fmap, which ignores entirely immutable leaves
      x in seen && return
      push!(seen, x)
    end
    push!(ps, x)
  elseif !_show_isleaf(x, m)
    if ismutable(x)  # a layer appearing twice is walked once
      x in seen && return
      push!(seen, x)
    end
    for c in _childvec(_show_trainable_walk(x, m))
      _show_trainables!(ps, seen, c)
    end
  end
  return
end

# Like `Functors.anymutable`, which decides what the cache of `fmap` remembers:
Base.@nospecializeinfer function _anymutable(x)::Bool
  ismutable(x) && return true
  T = typeof(x)
  for i in 1:nfields(x)
    fieldtype(T, i) == T && continue
    isdefined(x, i) && _anymutable(getfield(x, i)) && return true
  end
  return false
end

# Number and total length of all numeric arrays reached through `Functors.children`,
# trainable or not, counting tied arrays every time they appear:
Base.@nospecializeinfer function _show_arrays(x)
  acc = [0, 0]
  _show_arrays!(acc, x)
  return acc[1], acc[2]
end

Base.@nospecializeinfer function _show_arrays!(acc::Vector{Int}, x)
  if x isa AbstractArray{<:Number}
    acc[1] += 1
    acc[2] += length(x)::Int
  else
    m = _show_methods(typeof(x))
    _show_isleaf(x, m) && return
    for c in _childvec(_show_functor_children(x, m))
      _show_arrays!(acc, c)
    end
  end
  return
end

Base.@nospecializeinfer function _sum_length(ps::Vector{Any})
  n = 0
  for p in ps
    n += length(p)::Int
  end
  return n
end

# These give the same results as `Functors.children`, `Functors.isleaf`, `Optimisers.trainable`
# and `Optimisers.isnumeric`. But calling those compiles `Functors.functor(::Type{T}, x)` etc. for
# every new `T`, so instead, when `which` says that the default methods apply, we read the fields
# directly. The tests in test/layers/show.jl check that this agrees with Functors & Optimisers.

const _CHILDREN_DEFAULT = which(Functors.children, Tuple{Any})
const _FUNCTOR1_DEFAULT = which(Functors.functor, Tuple{Any})  # children(x) = functor(x)[1]
const _ISLEAF_DEFAULT = which(Functors.isleaf, Tuple{Any})
const _TRAINABLE_DEFAULT = which(Optimisers.trainable, Tuple{Any})
const _ISNUMERIC_DEFAULT = which(Optimisers.isnumeric, Tuple{Any})  # isnumeric(x) = false
const _FUNCTOR_DEFAULT = which(Functors.functor, Tuple{Any, Any})
const _FUNCTOR_ITSELF = (  # methods for which children(x) == x
  which(Functors.functor, Tuple{Type{Tuple{Int}}, Tuple{Int}}),
  which(Functors.functor, Tuple{Type{@NamedTuple{a::Int}}, @NamedTuple{a::Int}}),
  which(Functors.functor, Tuple{Type{Vector{Any}}, Vector{Any}}),
)

# Like the NamedTuple from the default `functor`, without making a new NamedTuple type:
struct _ShowFields
  names::Vector{Symbol}
  vals::Vector{Any}
end

Base.@nospecializeinfer function _which(f, tt::Type)
  try
    which(f, tt)
  catch  # e.g. ambiguity
    nothing
  end
end

# How to find the children of `x::T`:
@enum _ChildrenKind::Int8 begin
  _CALL_CHILDREN  # call `Functors.children(x)`
  _READ_FIELDS    # the default `functor` applies, so read the fields of `x`
  _ITSELF         # the children are `x` itself, e.g. a Tuple
end

# For each type, which of these default methods apply. Calling `which` is slow enough to matter
# for large models, so this is cached, and forgotten whenever the world age changes.
struct _ShowMethods
  children::_ChildrenKind
  isleaf::Bool     # `Functors.isleaf` is the default, which calls `children`
  trainable::Bool  # `Optimisers.trainable` is the default, which calls `children`
  isnumeric::Bool  # `Optimisers.isnumeric` is the default for non-arrays, which is `false`
end

const _SHOW_METHODS = IdDict{Any, _ShowMethods}()
const _SHOW_METHODS_WORLD = Ref(typemax(UInt))
const _SHOW_METHODS_LOCK = ReentrantLock()

Base.@nospecializeinfer function _show_methods(T::Type)
  world = Base.get_world_counter()
  lock(_SHOW_METHODS_LOCK)
  try
    if _SHOW_METHODS_WORLD[] != world
      empty!(_SHOW_METHODS)
      _SHOW_METHODS_WORLD[] = world
    end
    haskey(_SHOW_METHODS, T) && return _SHOW_METHODS[T]
    children = _CALL_CHILDREN
    if _which(Functors.children, Tuple{T}) === _CHILDREN_DEFAULT &&
        _which(Functors.functor, Tuple{T}) === _FUNCTOR1_DEFAULT
      m = _which(Functors.functor, Tuple{Type{T}, T})
      children = m === _FUNCTOR_DEFAULT ? _READ_FIELDS : m in _FUNCTOR_ITSELF ? _ITSELF : _CALL_CHILDREN
    end
    isleaf = _which(Functors.isleaf, Tuple{T}) === _ISLEAF_DEFAULT
    trainable = _which(Optimisers.trainable, Tuple{T}) === _TRAINABLE_DEFAULT
    isnumeric = _which(Optimisers.isnumeric, Tuple{T}) === _ISNUMERIC_DEFAULT
    return _SHOW_METHODS[T] = _ShowMethods(children, isleaf, trainable, isnumeric)
  finally
    unlock(_SHOW_METHODS_LOCK)
  end
end

# Each of these takes `m = _show_methods(typeof(x))`, to look that up once per node:

Base.@nospecializeinfer _show_functor_children(x) = _show_functor_children(x, _show_methods(typeof(x)))
Base.@nospecializeinfer function _show_functor_children(x, m::_ShowMethods)
  if m.children == _READ_FIELDS
    T = typeof(x)
    n = fieldcount(T)
    n == 0 && return ()  # this is Functors.NoChildren()
    names = Vector{Symbol}(undef, n)
    vals = Vector{Any}(undef, n)
    for i in 1:n
      names[i] = fieldname(T, i)
      vals[i] = getfield(x, i)
    end
    return _ShowFields(names, vals)
  elseif m.children == _ITSELF
    return x
  else
    return Functors.children(x)
  end
end

Base.@nospecializeinfer _show_isleaf(x) = _show_isleaf(x, _show_methods(typeof(x)))
Base.@nospecializeinfer function _show_isleaf(x, m::_ShowMethods)::Bool
  if !m.isleaf
    return Functors.isleaf(x)
  elseif m.children == _READ_FIELDS
    return fieldcount(typeof(x)) == 0
  elseif m.children == _ITSELF
    return x === ()
  else
    return Functors.children(x) === ()
  end
end

Base.@nospecializeinfer _show_trainable(x) = _show_trainable(x, _show_methods(typeof(x)))
Base.@nospecializeinfer function _show_trainable(x, m::_ShowMethods)
  return m.trainable ? _show_functor_children(x, m) : Optimisers.trainable(x)
end

# What `Optimisers.trainables` walks into, which differs from `trainable(x)` only for the
# old-style `trainable(x) = (x.a,)`, from which it keeps the children that are `in` this tuple:
Base.@nospecializeinfer function _show_trainable_walk(x, m::_ShowMethods)
  m.trainable && return _show_functor_children(x, m)
  tr = Optimisers.trainable(x)
  tr isa Tuple || return tr
  ch = Functors.children(x)
  ch isa NamedTuple || return tr
  out = Vector{Any}(undef, nfields(ch))
  for i in eachindex(out)
    c = getfield(ch, i)
    out[i] = c in tr ? c : nothing
  end
  return out
end

Base.@nospecializeinfer function _show_isnumeric(x, m::_ShowMethods)::Bool
  if x isa AbstractArray{<:Number}
    return Optimisers.isnumeric(x)
  else
    return !m.isnumeric && Optimisers.isnumeric(x)
  end
end

# utility functions

underscorise(n::Integer) =
  join(reverse(join.(reverse.(Iterators.partition(digits(n), 3)))), '_')

Base.@nospecializeinfer function _nan_show(io::IO, ps::Vector{Any})
  isempty(ps) && return
  allzero, somenan, someinf = true, false, false
  for p in ps
    m = _maxabs(p)
    if m === nothing  # not an array
      allzero = false
      continue
    end
    allzero &= iszero(m)::Bool
    somenan |= isnan(m)::Bool
    someinf |= isinf(m)::Bool
  end
  if allzero
    printstyled(io, "  (all zero)", color=:cyan)
  elseif somenan
    printstyled(io, "  (some NaN)", color=:red)
  elseif someinf
    printstyled(io, "  (some Inf)", color=:red)
  end
end

@specialize  # un-does @nospecialize at the top of this file

# This is compiled for each type of array, but not for each model. It is NaN if any element is,
# so that one reduction (on the GPU, one kernel) tells us whether all are zero, some NaN, or some Inf.
# Dense arrays are reshaped to vectors, so that matrices etc. don't need another kernel.
_maxabs(x::DenseArray{<:Number}) = _maxabs_(vec(x))
_maxabs(x::AbstractArray{<:Number}) = _maxabs_(x)
_maxabs(x::AbstractArray) = _maxabs_(Iterators.filter(y -> y isa Number, x))  # e.g. with missing
_maxabs(x) = nothing  # something else for which `Optimisers.isnumeric` is true
function _maxabs_(x)
  T = eltype(x)
  init = isconcretetype(T) ? zero(real(T)) : false  # e.g. Vector{Real}
  mapreduce(_absmax, max, x; init)
end
# Unlike `abs(z)`, this is NaN whenever `isnan(z)`, and doesn't overflow:
_absmax(x) = abs(x)
_absmax(z::Complex) = max(abs(real(z)), abs(imag(z)))

#=

julia> struct Tmp2; x; y; end;

julia> t = Tmp2([Dense(2,3), randn(3,4)'], (x=1:4, y=Dense(3,4), z=rand(3)))
Tmp2(Any[Dense(2 => 3), [-0.559390071462934 -0.6357914190386781 -0.8516823037180543; -2.187495592853204 -0.6807254521505784 -1.2334639710489697; -0.12790952072543338 -1.4672700459421741 1.3687526519721238; 0.5232171922680576 -1.012045481192333 1.4953790632112915]], (x = 1:4, y = Dense(3 => 4), z = [0.29222096031585143, 0.6562195256556428, 0.9741896713499167]))

julia> Chain(t)
Chain(
  Tmp2(
    [
      Dense(2 => 3),                    # 9 parameters
      4×3 Adjoint{Float64,...},         # 12 parameters
    ],
    (;
      x = 4-element UnitRange{Int64},
      y = Dense(3 => 4),                # 16 parameters
      z = 3-element Vector{Float64},    # 3 parameters
    ),
  ),
)         # Total: 6 trainable arrays, 40 parameters,
          # plus 1 non-trainable, 4 parameters, summarysize 620 bytes.


julia> Flux.@layer Tmp2

julia> t
Tmp2(
  [
    Dense(2 => 3),                      # 9 parameters
    4×3 Adjoint{Float64,...},           # 12 parameters
  ],
  4-element UnitRange{Int64},
  Dense(3 => 4),                        # 16 parameters
  3-element Vector{Float64},            # 3 parameters
)         # Total: 6 trainable arrays, 40 parameters,
          # plus 1 non-trainable, 4 parameters, summarysize 620 bytes.

julia> Chain(t)
Chain(
  Tmp2(
    [
      Dense(2 => 3),                    # 9 parameters
      4×3 Adjoint{Float64,...},         # 12 parameters
    ],
    4-element UnitRange{Int64},
    Dense(3 => 4),                      # 16 parameters
    3-element Vector{Float64},          # 3 parameters
  ),
)         # Total: 6 trainable arrays, 40 parameters,
          # plus 1 non-trainable, 4 parameters, summarysize 620 bytes.
=#
