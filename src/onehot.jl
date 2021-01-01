using Base: @_noinline_meta, @_inline_meta
using Core: is_top_bit_set
using Core.Intrinsics: bitcast, trunc_int, sext_int, zext_int, sle_int, eq_int, and_int

abstract type AbstractOneHotArray{N} <: AbstractArray{Bool,  N} end

# onehot erorr
struct OneHotEncodeError <: Exception
  K
  val
  OneHotEncodeError(@nospecialize(K), @nospecialize(val)) = (@_noinline_meta; new(K, val))
end

function Base.showerror(io::IO, e::OneHotEncodeError)
  print(io, "OneHotEncodeError: cannot encode ")
  print(io, e.val)
  print(io, " with OneHot{")
  print(io, e.K)
  print(io, '}')
end

throw_onehotencode_error(K, val) = (@_noinline_meta; throw(OneHotEncodeError(K, val)))

# onehot encode
primitive type OneHot{K} <: AbstractOneHotArray{1} 32 end

OneHot(k) = OneHot{UInt32(k)}
OneHot{K}(x) where K = convert(OneHot(K), x)
OneHot(k, x) = OneHot{k}(x)

onehotsize(::OneHot{K}) where K = Int(K)

# array interface

Base.size(o::OneHot) = (onehotsize(o),)
function Base.getindex(o::OneHot, i::I) where {I<:Integer}
  @boundscheck checkbounds(o, i)
  return convert(I, o) == i
end

Base.getindex(o::OneHot, i::Colon) = o

Base.argmax(o::OneHot) = Int(o)

# printing

function Base.showarg(io::IO, x::OneHot, toplevel)
  toplevel || print(io, "::")
  join(io, ["OneHot{", onehotsize(x), '}'])
end

# convert

Base.UInt32(o::OneHot) = bitcast(UInt32, o)
Base.UInt64(o::OneHot) = zext_int(UInt64, o)
Base.Int32(o::OneHot) = bitcast(Int32, o)
Base.Int64(o::OneHot) = zext_int(Int64, o)

Base.convert(::Type{Any}, o::OneHot) = o
Base.convert(::Type{OneHot{K}}, o::OneHot{K}) where {K} = o
Base.convert(::Type{UInt32},  o::OneHot) = UInt32(o)
Base.convert(::Type{To}, o::OneHot) where {To} = convert(To, convert(UInt32, o))

Base.convert(ot::Type{OneHot{K}}, x::Core.BuiltinInts) where {K} = toOneHot(ot, x)

# zero

Base.zero(o::O) where {O<:OneHot} = toOneHot(O, 0x00000000)
Base.zero(::Type{<:OneHot{K}}) where K = OneHot(K, 0)
Base.iszero(o::O) where {O<:OneHot} = iszero(convert(UInt32, o))

# number

Base.typemin(::Type{<:OneHot{K}}) where K = OneHot(K, 0)
Base.typemax(::Type{<:OneHot{K}}) where K = OneHot(K, K)

# bit-op

function check_onehot_top_bit(::Type{OneHot{K}}, x) where {K}
  @_inline_meta
  is_top_bit_set(x) && throw_onehotencode_error(K, x)
  x
end

function check_onehot_encode(ot::Type{OneHot{K}}, x) where {K}
  @_inline_meta
  sle_int(x, K) || throw_onehotencode_error(K, x)
  bitcast(ot, x)
end

function checked_onehot_trunc_sint(ot::Type{OneHot{K}}, x::From) where {K, From}
  @_inline_meta
  y = trunc_int(UInt32, x)
  back = sext_int(From, y)
  eq_int(x, back) || throw_onehotencode_error(K, x)
  check_onehot_encode(ot, y)
end

function checked_onehot_trunc_uint(ot::Type{OneHot{K}}, x::From) where {K, From}
  @_inline_meta
  y = trunc_int(UInt32, x)
  back = zext_int(From, y)
  eq_int(x, back) || throw_onehotencode_error(K, x)
  check_onehot_encode(ot, y)
end

toOneHot(ot::Type{OneHot{K}}, x::Int8) where {K} = check_onehot_encode(ot, sext_int(UInt32, check_onehot_top_bit(ot, x)))
toOneHot(ot::Type{OneHot{K}}, x::Int16) where {K} = check_onehot_encode(ot, sext_int(UInt32, check_onehot_top_bit(ot, x)))
toOneHot(ot::Type{OneHot{K}}, x::Int32) where {K} = check_onehot_encode(ot, bitcast(UInt32, check_onehot_top_bit(ot, x)))
toOneHot(ot::Type{OneHot{K}}, x::Int64) where {K} = checked_onehot_trunc_sint(ot, check_onehot_top_bit(ot, x))
toOneHot(ot::Type{OneHot{K}}, x::Int128) where {K} = checked_onehot_trunc_sint(ot, check_onehot_top_bit(ot, x))
toOneHot(ot::Type{OneHot{K}}, x::UInt8) where {K} = check_onehot_encode(ot, zext_int(UInt32, x))
toOneHot(ot::Type{OneHot{K}}, x::UInt16) where {K} = check_onehot_encode(ot, zext_int(UInt32, x))
toOneHot(ot::Type{OneHot{K}}, x::UInt32) where {K} = check_onehot_encode(ot, x)
toOneHot(ot::Type{OneHot{K}}, x::UInt64) where {K} = checked_onehot_trunc_uint(ot, x)
toOneHot(ot::Type{OneHot{K}}, x::UInt128) where {K} = checked_onehot_trunc_uint(ot, x)
toOneHot(ot::Type{OneHot{K}}, x::Bool) where {K} = and_int(zext_int(ot, x), toOneHot(ot, 0x1))

# onehot array
struct OneHotArray{K, N, var"N+1", A<:AbstractArray{OneHot{K}, N}} <: AbstractOneHotArray{var"N+1"}
  onehots::A
end

OneHotArray(onehots::A) where {K, A<:AbstractArray{OneHot{K}}} = OneHotArray{K, ndims(onehots), ndims(onehots)+1, A}(onehots)
OneHotArray{K}(indices::A) where {K, A<:AbstractArray{<:Integer}} = OneHotArray(K, indices)
OneHotArray(k, xs) = OneHotArray(OneHot(k).(xs))

OneHotArray{K, N}(xs::AbstractArray{T, N}) where {K, N, T} = OneHotArray{K, N, N+1}(xs)
function OneHotArray{K, N, var"N+1"}(xs::AbstractArray{T, N}) where {K, N, var"N+1", T}
  @assert N+1 == var"N+1"
  OneHotArray(K, xs)
end

const OneHotVector{K} = OneHot{K}
const OneHotMatrix{K} = OneHotArray{K, 1}

onehotsize(::OneHotArray{K}) where K = Int(K)

# array interface
Base.size(oa::OneHotArray{K}) where K = (onehotsize(oa), size(oa.onehots)...)

function Base.getindex(oa::OneHotArray{K, N}, i, is::Vararg{Int, N}) where {K, N}
  @boundscheck checkbounds(oa, i, is...)
  oa.onehots[is...][i]
end

function Base.getindex(oa::OneHotArray{K, N}, i::Colon, is::Vararg{Int, N}) where {K, N}
  @boundscheck checkbounds(oa, i, is...)
  oa.onehots[is...]
end

function Base.getindex(oa::OneHotArray{K}, i::Colon, is...) where {K}
  @boundscheck checkbounds(oa, i, is...)
  OneHotArray(oa.onehots[is...])
end

function Base.getindex(oa::OneHotArray{K}, i::Integer, is::Vararg{Colon, N}) where {K, N}
  @boundscheck checkbounds(oa, i, is...)
  map(x->x[i], oa.onehots)
end

Base.similar(o::OneHotArray, ::Type{T}, dims::Dims{N}) where {T, N} = similar(o.onehots, T, dims)

# printing

function Base.summary(io::IO, oa::OneHotArray)
  join(io, size(oa), 'x')
  join(io, [" OneHotArray{", onehotsize(oa), ", ", ndims(oa), ", "])
  Base.showarg(io, oa.onehots, true)
  print(io, "}")
end

# cat

Base.vcat(xss::OneHot{K}...) where K = cat(xss...; dims=Val(1))
Base.hcat(xss::OneHot{K}...) where K = cat(xss...; dims=Val(2))

function Base.cat(xss::OneHot{K}...; dims) where K
  isone(::Val{V}) where V = isone(V)
  isone(v) = Base.isone(v)
  if isone(dims)
    @warn "concat OneHot{$K} along dimension 1."
    Base._cat(Val(1), xss...)
  else
    predecessor(::Val{V}) where V = Val(V-1)
    predecessor(v) = Val(v - 1)
    yss = reshape(collect(xss), reverse(Base.rdims(predecessor(dims), axes(xss))))
    OneHotArray(yss)
  end
end

Base.vcat(xss::OneHotArray{K}...) where K = cat(xss...; dims=Val(1))
Base.hcat(xss::OneHotArray{K}...) where K = cat(xss...; dims=Val(2))

function Base.cat(xss::OneHotArray{K}...; dims) where K
  isone(::Val{V}) where V = isone(V)
  isone(v) = Base.isone(v)
  if isone(dims)
    @warn "concat OneHotArray{$K} along dimension 1."
    Base._cat(Val(1), xss...)
  else
    predecessor(::Val{V}) where V = Val(V-1)
    predecessor(v) = v - 1
    sdims = predecessor(dims)
    xidss = map(xs->xs.onehots, xss)
    ret = cat(xidss...; dims=sdims)
    OneHotArray(ret)
  end
end

# reshape

"reshape the onehots"
function ohreshape(parent::OneHotArray{K}, dims) where K
  onehots = parent.onehots
  OneHotArray(reshape(onehots, dims))
end

function Base.reshape(parent::OneHotArray{K}, dims::Dims) where K
  isequal(prod(dims), length(parent)) || throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(length(parent))"))
  return isequal(K, first(dims)) ?
    ohreshape(parent, Base.tail(dims)) :
    Base._reshape(parent, dims)
end

function Base.reshape(parent::OneHotArray{K}, dims::Tuple{Vararg{Union{Colon, Int}}}) where K
  rdims = Base._reshape_uncolon(parent, dims)
  return isequal(K, first(rdims)) ?
    ohreshape(parent, Base.tail(rdims)) :
    Base._reshape(parent,
                  rdims)
end

# gpu
import Adapt: adapt, adapt_structure
adapt_structure(T, oa::OneHotArray) = OneHotArray(adapt(T, oa.onehots))

import .CUDA
import .CUDA: CuArray, CuArrayStyle, @allowscalar

Base.BroadcastStyle(::Type{<: OneHotArray{K, N, var"N+1", A}}) where {K, N, var"N+1", A <: CuArray} = CuArrayStyle{var"N+1"}()

# avoid scalar operation by broadcasting assigment
CuArray(o::OneHotArray) = CuArray{Bool}(o)
function CuArray{F}(o::OneHotArray{K, N, var"N+1", A}) where {F, K, N, var"N+1", A <: CuArray}
  dest = similar(o, F)
  dest .= o
  return dest
end

using Base.Cartesian

function Base.findmax(o::OneHotArray{K, N, var"N+1", A}; dims=:) where {K, N, var"N+1", A <: CuArray}
  if dims == Colon()
    return (true, argmax(o, dims=dims))
  elseif isone(dims)
    return (CUDA.ones(Bool, size(o.onehots)), argmax(o, dims=dims))
  else
    a = CuArray{Bool}(o)
    return findmax(a, dims=dims)
  end
end

function Base.argmax(o::OneHotArray{K, N, var"N+1", A}; dims=:) where {K, N, var"N+1", A <: CuArray}
  if dims == Colon()
    n = findfirst(!isequal(zero(eltype(o.onehots))), o.onehots)
    x = @allowscalar o.onehots[n]
    return CartesianIndex(Int(x), n)
  elseif isone(dims)
    return map((x, i)->CartesianIndex(Int(x), i), o.onehots, CartesianIndices(o.onehots)) |> Base.Fix2(reshape, (1, size(o.onehots)...))
  else
    return findmax(o, dims=dims)[2]
  end
end

# api

batch(xs::AbstractVector{<:OneHot}) = OneHotArray(xs)

"""
    onehot(l, labels[, unk])

Return a `OneHot` where only first occourence of `l` in `labels` is `1` and
all other elements are `0`.

If `l` is not found in labels and  `unk` is present, the function returns
`onehot(unk, labels)`; otherwise the function raises an error.

# Examples
```jldoctest
julia> Flux.onehot(:b, [:a, :b, :c])
3-element Flux.OneHot{3}:
 0
 1
 0

julia> Flux.onehot(:c, [:a, :b, :c])
3-element Flux.OneHot{3}:
 0
 0
 1
```
"""
function onehot(l, labels)
  i = something(findfirst(isequal(l), labels), 0)
  i > 0 || error("Value $l is not in labels")
  OneHot(length(labels), i)
end

function onehot(l, labels, unk)
  i = something(findfirst(isequal(l), labels), 0)
  i > 0 || return onehot(unk, labels)
  OneHot(length(labels), i)
end

"""
    onehotbatch(ls, labels[, unk...])

Return a `OneHotMatrix` where `k`th column of the matrix is `onehot(ls[k], labels)`.

If one of the input labels `ls` is not found in `labels` and `unk` is given,
return [`onehot(unk, labels)`](@ref) ; otherwise the function will raise an error.

# Examples
```jldoctest
julia> Flux.onehotbatch([:b, :a, :b], [:a, :b, :c])
3×3 Flux.OneHotArray{3, 2, Array{Flux.OneHot{0x00000003},1}}:
 0  1  0
 1  0  1
 0  0  0
```
"""
onehotbatch(ls, labels, unk...) =
  OneHotMatrix{length(labels)}([onehot(l, labels, unk...) for l in ls])

"""
    onecold(y[, labels = 1:length(y)])

Inverse operations of [`onehot`](@ref).

# Examples
```jldoctest
julia> Flux.onecold([true, false, false], [:a, :b, :c])
:a

julia> Flux.onecold([0.3, 0.2, 0.5], [:a, :b, :c])
:c
```
"""
function onecold(y::AbstractArray, labels = 1:size(y, 1))
  @assert size(y, 1) == length(labels)
  indices = _onecold(y)
  if labels == 1:size(y, 1)
    return indices
  else
    if isbits(labels)
      xs = indices
    else
      xs = collect(indices) # non-bit type cannot be handled by CUDA
    end
    return map(xi -> labels[xi], xs)
  end
end

onecold(y::AbstractVector, labels = 1:length(y)) = labels[Base.argmax(y)]

_onecold(y::AbstractArray) = dropdims(map(x->Int32(x[1]), Base.argmax(y, dims=1)), dims=1)
_onecold(y::OneHotArray) = convert(AbstractArray{Int32}, y.onehots)

# AD
@nograd OneHot, OneHotArray, onecold, onehot, onehotbatch

import Base: *
function Base.:(*)(A::AbstractMatrix, B::AbstractOneHotArray)
  size(A, 2) == onehotsize(B) || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $(onehotsize(B))"))
  A[:, onecold(B)]
end
