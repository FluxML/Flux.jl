import Flux: Recur, _truncate, prefor, glorot_uniform, gate

function identities(n::Int64)
    fill(identity, n)
end

# reduce/merge modes in keras: 'sum', 'mul', 'concat', 'ave', None
# see https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L333

function concat(xs)
    vcat(xs...)
end

function mul(xs)
    D = xs[1]
    for i in 2:length(xs)
        D = D .* xs[i]
    end
    D
end

function mean(xs)
    D = sum(xs) ./ length(xs)
    D
end


mutable struct Parallel  #{M<:Recur}
    layers::Vector
    map::Vector{Function}
    inv::Vector{Function}
    reduce::Function
end

function Parallel(layers::Vector)
    Parallel(layers, identities(length(layers)), identities(length(layers)), concat)
end

function Parallel(
    layers::Vector;
    map::Dict{Int64,Function} = Dict{Int64,Function}(),
    reduce::Function = concat) # where M<:Recur

    # TODO: throw error for min length for layers - 1 or 2?

    mappings::Vector{Function} = identities(length(layers))
    inverses::Vector{Function} = copy(mappings)
    for (k,v) in map
        mappings[k] = v
        inverses[k] = v
    end

    return Parallel(layers, mappings, inverses, reduce)
end

function (p::Parallel)(xs)
    # NOTE: The implementation of the mapping is acutally sequential and not parallel.
    #       How to parallelize for cpu() and gpu() ia an open question to me, as `Base.pmap` does not exist anymore and 
    #       `Threads.@threads` does not seem to be a good idea neither.

    # double reverse, analog to `Flux.flip` but without broadcast; see: recurrent.jl#flip(f, xs)
    # y = map^-1(f(map(x))) or map(x) |> f |> map^-1
    apply(l) = p.inv[l](p.layers[l](p.map[l](xs)))
    
    # implicit mapping
    # Z = map(l -> apply(l), eachindex(p.layers))

    # explicit mapping - define type and size of Z    
    first = apply(1)
    Z = Vector{typeof(first)}(UndefInitializer(), length(p.layers))
    for l in eachindex(p.layers)
        if l == 1
            Z[l] = first
        else
            Z[l] = apply(l)
        end
    end

    p.reduce(Z)
end

# NOTE: Instead of generating `Flux.children` and `Flux.mapchildren` with `@treelike` macro, they are defined 
#       explicity, as `@treelike Parallel layers` is considerd not treelike: `error("@treelike T (a, b)")`
Flux.children(p::Parallel) = p.layers
Flux.mapchildren(f, p::Parallel) = Parallel(f.(p.layers), p.map, p.inv, p.reduce)

Base.show(io::IO, m::Parallel) = print(io, "Parallel(", m.layers, ", ", m.map, ", ", m.inv, ", ", m.reduce, ")")

function _truncate_parallel(x)
    if x isa Recur
        x.state = _truncate(x.state)
    elseif x isa Parallel
        for layer in x.layers
            _truncate_parallel(layer)
        end
    end
end

function truncate_parallel!(m)
    prefor(_truncate_parallel, m)
end

function _reset_parallel(x)
    if x isa Recur
        x.state = x.init
    elseif x isa Parallel
        for recur in x.layers
            _reset_parallel(recur)
        end
    end
end

function reset_parallel!(m)
    prefor(_reset_parallel, m)
end

function Base.reverse(M::Flux.OneHotMatrix{Array{Flux.OneHotVector,1}})
    # Flux.OneHotMatrix(M.height, reverse(M.data))
    # M.data = reverse(M.data)
    reverse(M)
end

function Base.reverse(v::Flux.OneHotVector)
    v
end

function Base.reverse(ta::TrackedArray)
    if length(size(ta.data)) == 2
        # TODO: does this need to be param() or params() or how to propagate the tracking?
        # reverse(ta.data, dims=2)
        # param(reverse(ta.data, dims=2))
        reverse(ta)
    else
        ta
    end
end

function Base.reverse(b::Bool)
    b
end

function Base.reverse(x::Number)
    x
end

# see:
#  "SPEECH RECOGNITION WITH DEEP RECURRENT NEURAL NETWORKS" https://arxiv.org/pdf/1303.5778.pdf
#  "Bidirectional LSTM-CRF Models for Sequence Tagging" https://arxiv.org/pdf/1508.01991.pdf
#  Bidirectional layer of Keras: https://github.com/keras-team/keras/blob/05d713504852b490afcf2607aea1ce923e93ecfe/keras/layers/wrappers.py#L333
function Bi(layer::Recur, reduce::Function = concat)
    map = Dict{Int64,Function}(2 => reverse)
    Parallel([layer, deepcopy(layer)], map=map, reduce=reduce)
end

function BiLSTM(in::Int, out::Int; reduce::Function = concat)
    Bi(LSTM(in, out), reduce)
end

function BiPLSTM(in::Int, out::Int; reduce::Function = concat)    
    Bi(PLSTM(in, out), reduce)
end
