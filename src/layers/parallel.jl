import Flux: Recur, _truncate, prefor, glorot_uniform, gate

function identities(n::Int64)
    fill(identity, n)
end

# reduce/merge modes in keras: 'sum', 'mul', 'concat', 'ave', None
# see https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L333

function concat(xs)
    # implicit concatenation
    return vcat(xs...)

    # explcit concatenation - preallocated size
    # n = length(xs)
    # m = length(xs[1])
    # concated = Array{eltype(xs[1].data)}(UndefInitializer(), n * m)
    # for i in 1:n
    #     concated[(i-1)*m+1:i*m] = xs[i].data
    # end
    # return param(concated)
end

function mul(xs)
    D = xs[1]
    for i in 2:length(xs)
        D = D .* xs[i]
    end
    D
end

# function mean(xs)
#     D = sum(xs) ./ length(xs)
#     D
# end


mutable struct Parallel
    layers::Vector
    map::Vector{Function}
    inv::Vector{Function}
    reduce::Function
end


function Parallel(
    layers::Vector;
    map::Dict{Int64,Function} = Dict{Int64,Function}(),
    inv::Dict{Int64,Function} = Dict{Int64,Function}(),
    reduce::Function = concat)

    mappings::Vector{Function} = identities(length(layers))
    for (k,v) in map
        mappings[k] = v
    end

    inverses::Vector{Function} = identities(length(layers))
    for (k,v) in inv
        inverses[k] = v
    end

    return Parallel(layers, mappings, inverses, reduce)
end

function (p::Parallel)(xs)
    # NOTE: The implementation of the mapping is acutally sequential and not parallel.
    #       How to parallelize for cpu() and gpu() ia an open question to me, as `Base.pmap` does not exist anymore and 
    #       `Threads.@threads` does not seem to be a good idea neither.

    # double reverse; analog to `Flux.flip`, but without broadcast; see: recurrent.jl#flip(f, xs)
    # y = map^-1(f(map(x))) or map(x) |> f |> map^-1
    apply(l) = p.inv[l](p.layers[l](p.map[l](xs)))
    
    # implicit mapping
    Z = map(l -> apply(l), eachindex(p.layers))

    # explicit mapping - preallocated size
    # first = apply(1)
    # Z = Vector{typeof(first)}(UndefInitializer(), length(p.layers))
    # for l in eachindex(p.layers)
    #     if l == 1
    #         Z[l] = first
    #     else
    #         Z[l] = apply(l)
    #     end
    # end

    p.reduce(Z)
end

# NOTE: Instead of generating `Flux.children` and `Flux.mapchildren` with `@treelike` macro, they are defined 
#       explicity, as `@treelike Parallel layers` is considerd not treelike: `error("@treelike T (a, b)")`
Flux.children(p::Parallel) = p.layers
Flux.mapchildren(f, p::Parallel) = Parallel(f.(p.layers), p.map, p.inv, p.reduce)

function Base.show(io::IO, m::Parallel)
    print(io, "Parallel(\n")
    print(io, "  ", m.layers, ",\n")
    print(io, "     map = ", m.map, ",\n")
    print(io, "     inv = ", m.inv, ",\n")
    print(io, "  reduce = ", m.reduce, "\n")
    print(io, ")")
end


"""
    reverse(M)
    
Reverse the input vector for each batch.

# Examples
```julia-repl
julia> M = onehotbatch([:a, :a, :b, :b], [:a, :b, :c])
3×4 Flux.OneHotMatrix{Array{Flux.OneHotVector,1}}:
  true   true  false  false
 false  false   true   true
 false  false  false  false

julia> reverse(M)
3×4 Flux.OneHotMatrix{Array{Flux.OneHotVector,1}}:
 false  false   true   true
  true   true  false  false
 false  false  false  false
```
"""
function Base.reverse(M::Flux.OneHotMatrix{Array{Flux.OneHotVector,1}})
    Flux.OneHotMatrix(M.height, reverse(M.data))
end

"""
    reverse(v)

Return the identity of the one-hot vector.
"""
function Base.reverse(v::Flux.OneHotVector)
    v
end

"""
    reverse(x)

Reverse a tracked array or matrix.

# Examples
```julia-repl

julia> a = param([1, 2, 3, 4, 5, 6])
Tracked 6-element Array{Float64,1}:
 1.0
 2.0
 3.0
 4.0
 5.0
 6.0

julia> reverse(a)
Tracked 6-element Array{Float64,1}:
 6.0
 5.0
 4.0
 3.0
 2.0
 1.0

julia> T = param([1 2 3; 4 5 6])
Tracked 2×3 Array{Float64,2}:
 1.0  2.0  3.0
 4.0  5.0  6.0

julia> reverse(T)
Tracked 2×3 Array{Float64,2}:
 3.0  2.0  1.0
 6.0  5.0  4.0
```
"""
function Base.reverse(ta::Flux.TrackedArray; dims=2)
    if length(size(ta.data)) == 2
        param(reverse(ta.data, dims=dims))
    else
        param(reverse(ta.data))
    end
end

"""
    reverse(v)

Reversing a tracked number returns the identity of the number.
"""
function Base.reverse(x::Flux.Tracker.TrackedReal)
    x
end


# see:
#  "SPEECH RECOGNITION WITH DEEP RECURRENT NEURAL NETWORKS" https://arxiv.org/pdf/1303.5778.pdf
#  "Bidirectional LSTM-CRF Models for Sequence Tagging" https://arxiv.org/pdf/1508.01991.pdf
#  Bidirectional layer of Keras: https://github.com/keras-team/keras/blob/05d713504852b490afcf2607aea1ce923e93ecfe/keras/layers/wrappers.py#L333
function Bi(layer::Recur, reduce::Function = concat)
    mapping = Dict{Int64,Function}(2 => reverse)
    Parallel([layer, deepcopy(layer)], map=mapping, inv=mapping, reduce=reduce)
end

function BiLSTM(in::Int, out::Int, reduce::Function = concat)
    Bi(LSTM(in, out), reduce)
end

function BiPLSTM(in::Int, out::Int, reduce::Function = concat)    
    Bi(PLSTM(in, out), reduce)
end
