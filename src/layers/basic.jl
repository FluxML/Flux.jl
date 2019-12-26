"""
    Chain(layers...)

Chain multiple layers / functions together, so that they are called in sequence
on a given input.

```julia
m = Chain(x -> x^2, x -> x+1)
m(5) == 26

m = Chain(Dense(10, 5), Dense(5, 2))
x = rand(10)
m(x) == m[2](m[1](x))
```

`Chain` also supports indexing and slicing, e.g. `m[2]` or `m[1:end-1]`.
`m[1:3](x)` will calculate the output of the first three layers.
"""
struct Chain{T<:Tuple}
  layers::T
  Chain(xs...) = new{typeof(xs)}(xs)
end

@forward Chain.layers Base.getindex, Base.length, Base.first, Base.last,
  Base.iterate, Base.lastindex

functor(c::Chain) = c.layers, ls -> Chain(ls...)

applychain(::Tuple{}, x) = x
applychain(fs::Tuple, x) = applychain(tail(fs), first(fs)(x))

(c::Chain)(x) = applychain(c.layers, x)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end


# This is a temporary and naive implementation
# it might be replaced in the future for better performance
# see issue https://github.com/FluxML/Flux.jl/issues/702
# Johnny Chen -- @johnnychen94
# only slightly changed to better handle interaction with Zygote @dsweber2
"""
    activations(c::Chain, input)
Calculate the forward results of each layers in Chain `c` with `input` as model input.
"""
function activations(c::Chain, input)
    extraChain(c.layers, input)
end

function extraChain(fs::Tuple, x)
    res = first(fs)(x)
    return (res, extraChain(Base.tail(fs), res)...)
end

extraChain(::Tuple{}, x) = ()



"""
    Dense(in::Integer, out::Integer, σ = identity)

Creates a traditional `Dense` layer with parameters `W` and `b`.

    y = σ.(W * x .+ b)

The input `x` must be a vector of length `in`, or a batch of vectors represented
as an `in × N` matrix. The out `y` will be a vector or batch of length `out`.

```julia
julia> d = Dense(5, 2)
Dense(5, 2)

julia> d(rand(5))
Tracked 2-element Array{Float64,1}:
  0.00257447
  -0.00449443
```
"""
struct Dense{F,S,T}
  W::S
  b::T
  σ::F
end

Dense(W, b) = Dense(W, b, identity)

function Dense(in::Integer, out::Integer, σ = identity;
               initW = glorot_uniform, initb = zeros)
  return Dense(initW(out, in), initb(out), σ)
end

@functor Dense

function (a::Dense)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  σ.(W*x .+ b)
end

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

# Try to avoid hitting generic matmul in some simple cases
# Base's matmul is so slow that it's worth the extra conversion to hit BLAS
(a::Dense{<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::Dense{<:Any,W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

"""
    Diagonal(in::Integer)

Creates an element-wise linear transformation layer with learnable
vectors `α` and `β`:

    y = α .* x .+ β

The input `x` must be a array where `size(x, 1) == in`.
"""
struct Diagonal{T}
  α::T
  β::T
end

Diagonal(in::Integer; initα = ones, initβ = zeros) =
  Diagonal(initα(in), initβ(in))

@functor Diagonal

function (a::Diagonal)(x)
  α, β = a.α, a.β
  α.*x .+ β
end

function Base.show(io::IO, l::Diagonal)
  print(io, "Diagonal(", length(l.α), ")")
end


"""
    Maxout(over)

`Maxout` is a neural network layer, which has a number of internal layers,
which all have the same input, and the maxout returns the elementwise maximium
of the internal layers' outputs.

Maxout over linear dense layers satisfies the univeral approximation theorem.

Reference:
Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, and Yoshua Bengio.
2013. Maxout networks.
In Proceedings of the 30th International Conference on International Conference on Machine Learning - Volume 28 (ICML'13),
Sanjoy Dasgupta and David McAllester (Eds.), Vol. 28. JMLR.org III-1319-III-1327.
https://arxiv.org/pdf/1302.4389.pdf
"""
struct Maxout{FS<:Tuple}
    over::FS
end

"""
    Maxout(f, n_alts)

Constructs a Maxout layer over `n_alts` instances of  the layer given  by `f`.
The function takes no arguement and should return some callable layer.
Conventionally this is a linear dense layer.

For example the following example which
will construct a `Maxout` layer over 4 internal dense linear layers,
each identical in structure (784 inputs, 128 outputs).
```julia
    insize = 784
    outsize = 128
    Maxout(()->Dense(insize, outsize), 4)
```
"""
function Maxout(f, n_alts)
  over = Tuple(f() for _ in 1:n_alts)
  return Maxout(over)
end

@functor Maxout

function (mo::Maxout)(input::AbstractArray)
    mapreduce(f -> f(input), (acc, out) -> max.(acc, out), mo.over)
end

"""
    SkipConnection(layers, connection)

Creates a Skip Connection, of a layer or `Chain` of consecutive layers
plus a shortcut connection. The connection function will combine the result of the layers
with the original input, to give the final output.

The simplest 'ResNet'-type connection is just `SkipConnection(layer, +)`,
and requires the output of the layers to be the same shape as the input.
Here is a more complicated example:
```
m = Conv((3,3), 4=>7, pad=(1,1))
x = ones(5,5,4,10);
size(m(x)) == (5, 5, 7, 10)

sm = SkipConnection(m, (mx, x) -> cat(mx, x, dims=3))
size(sm(x)) == (5, 5, 11, 10)
```
"""
struct SkipConnection
  layers
  connection  # user can pass arbitrary connections here, such as (a,b) -> a + b
end

@functor SkipConnection

function (skip::SkipConnection)(input)
  skip.connection(skip.layers(input), input)
end

function Base.show(io::IO, b::SkipConnection)
  print(io, "SkipConnection(", b.layers, ", ", b.connection, ")")
end

"""
    GroupedConvolutions(connection, paths..., split=false)

Creates a group of convolutions from a set of layers or chains of consecutive layers.
Proposed in [Alexnet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ).

The connection function will combine the results of each paths, to give the final output.
If split is false, each path acts on all feature maps of the input.
If split is true, the feature maps of the input are evenly distributed across all paths.

Data should be stored in WHCN order (width, height, # channels, # batches).
In other words, a 100×100 RGB image would be a `100×100×3×1` array,
and a batch of 50 would be a `100×100×3×50` array.

`GroupedConvolutions` also supports indexing and slicing, e.g. `group[2]` or `group[1:end-1]` and `group[1:3]` will return the first three paths.

The names of the variables are consistent accross all examples:
`i` stands for input,
`a` and `b`, `c`, and `d` are `Chains`,
`g` represents a `GroupedConvolutions`,
`s` is a `SkipConnection`,
and `o` is the output.

Examples A, B, and C show how to use grouped convolutions in practice for [ResNeXt](https://arxiv.org/abs/1611.05431).
Batch Normalization and ReLU activations are left out for simplicity.

**Example A**: ResNeXt block without splitting.
```
i   = randn(7,7,256,16)
a() = Chain(Conv((1,1), 256=>4  , pad=(0,0)),
            Conv((3,3), 4  =>4  , pad=(1,1)),
            Conv((1,1), 4  =>256, pad=(0,0)))
g   = GroupedConvolutions(+, [a() for _ = 1:32]..., split=false)
s   = SkipConnection(g, +)
o   = s(i)
```

**Example B**: ResNeXt block without splitting and early concatenation.
```
i   = randn(7,7,256,16)
a() = Chain(Conv((1,1), 256=>4, pad=(0,0)),
            Conv((3,3), 4  =>4, pad=(1,1)))
b   = Chain(GroupedConvolutions((results...) -> cat(results..., dims=3), [a() for _ = 1:32]..., split=false),
            Conv((1,1), 128=>256, pad=(0,0)))
s   = SkipConnection(b, +)
o   = s(i)
```

**Example C**: ResNeXt block with splitting (and concatentation).
```
i = randn(7,7,256,16)
b = Chain(Conv((1,1), 256=>128, pad=(0,0)),
          GroupedConvolutions((results...) -> cat(results..., dims=3), [Conv((3,3), 4=>4, pad=(1,1)) for _ = 1:32]..., split=true),
          Conv((1,1), 128=>256, pad=(0,0)))
s = SkipConnection(b, +)
o = s(i)
```

Example D shows how to use grouped convolutions in practice for [Inception v1 (GoogLeNet)](https://research.google/pubs/pub43022/).

**Example D**: Inception v1 (GoogLeNet) block
(The numbers used in this example come from Inception block 3a.)
```
i = randn(28,28,192,16)
a =       Conv(   (1,1), 192=>64,  pad=(0,0), relu)
b = Chain(Conv(   (1,1), 192=>96,  pad=(0,0), relu), Conv((3,3), 96 =>128, pad=(1,1), relu))
c = Chain(Conv(   (1,1), 192=>16,  pad=(0,0), relu), Conv((5,5), 16 =>32 , pad=(2,2), relu))
d = Chain(MaxPool((3,3), stride=1, pad=(1,1)      ), Conv((1,1), 192=>32 , pad=(0,0), relu))
g = GroupedConvolutions((results...) -> cat(results..., dims=3), a, b, c, d, split=false)
o = g(i)
```
"""
struct GroupedConvolutions{T<:Tuple}
  connection  # user can pass arbitrary connections here, such as (a,b) -> a + b
  paths::T
  split::Bool
end

function GroupedConvolutions(connection, paths...; split::Bool=false)
   npaths = size(paths, 1)
   npaths > 1 || error("the number of paths (", npaths, ") is not greater than 1")
   GroupedConvolutions{typeof(paths)}(connection, paths, split)
 end

@forward GroupedConvolutions.paths (Base.getindex, Base.length, Base.first, Base.last, Base.iterate, Base.lastindex)

@functor GroupedConvolutions

function (group::GroupedConvolutions)(input)
  # get input size
  w::Int64, h::Int64, c::Int64, n::Int64 = size(input)
  # number of feature maps in input
  nmaps::Int64 = c
  # number of paths of the GroupedConvolution
  npaths::Int64 = size(group.paths, 1)

  if group.split == true
    # distributes the feature maps of the input over the paths
    # throw error if number of feature maps not divisible by number of paths
    mod(nmaps, npaths) == 0 || error("the number of feature maps in the input (", nmaps, ") is not divisible by the number of paths of the GroupedConvolution (", npaths, ")")

    # number of maps per path
    nmaps_per_path::Int64 = div(nmaps, npaths)

    # calculate the output for the grouped convolutions
    group.connection([path(input[:,:,_start_index(path_index, nmaps_per_path):_stop_index(path_index, nmaps_per_path),:]) for (path_index, path) in enumerate(group.paths)]...)
  else
    # uses the complete input for each path
    group.connection([path(input) for (path) in group.paths]...)
  end
end

# calculates the start index of the feature maps for a path
_start_index(path_index::Int64, nmaps_per_path::Int64) = (path_index - 1) * nmaps_per_path + 1
# calculates the stop index of the feature maps for a path
_stop_index(path_index::Int64, nmaps_per_path::Int64) = (path_index) * nmaps_per_path

function Base.show(io::IO, group::GroupedConvolutions)
  print(io, "GroupedConvolutions(", group.connection, ", ", group.paths, ", split=", group.split, ")")
end

"""
    ChannelShuffle(ngroups)

Creates a layer that shuffles feature maps by each time taking the first channel of each group.
Proposed in [ShuffleNet](https://arxiv.org/abs/1707.01083).

The number of channels in the input must be divisible by the square of the number of groups.
(Each group must have a multiple of the number of groups channels.)

Examples of channel shuffling:
* (4  channels, 2 groups) **ab,cd               -> ac,bd**
* (8  channels, 2 groups) **abcd,efgh           -> aebf,cgdh**
* (16 channels, 2 groups) **abcdefgh,ijklmnop   -> aibjckdl,emfngohp**
* (9  channels, 3 groups) **abc,def,ghi         -> adg,beh,cfi**
* (16 channels, 4 groups) **abcd,efgh,ijkl,mnop -> aeim,bfjn,cgko,dhlp**

Data should be stored in WHCN order (width, height, # channels, # batches).
In other words, a 100×100 RGB image would be a `100×100×3×1` array,
and a batch of 50 would be a `100×100×3×50` array.

The names of the variables are consistent accross all examples:
`i` stands for input,
`a`, `b`, and `c` are `Chains`,
`g` represents a `GroupedConvolutions`,
`s` is a `SkipConnection`,
and `o` is the output.

Examples A and B show how to use channel shuffling in practice for [ShuffleNet](https://arxiv.org/abs/1707.01083).
Batch Normalization and ReLU activations are left out for simplicity.

**Example A**: ShuffleNet v1 unit with stride=1.
(The numbers used in this example come from stage 2 and using 2 groups.)
```
i  = randn(28,28,200,16)
c  = Chain(GroupedConvolutions(+, [Conv((1,1), 200=>64, pad=(0,0)) for _ in 1:2]..., split=false),
           ChannelShuffle(2),
           DepthwiseConv((3,3), 64=>64, pad=(1,1), stride=(1,1)),
           GroupedConvolutions(+, [Conv((1,1), 64=>200, pad=(0,0)) for _ in 1:2]..., split=false))
s  = SkipConnection(c, +)
o  = s(i)
```

**Example B**: ShuffleNet v1 unit with stride=2.
(The numbers used in this example come from stage 2 and using 2 groups.)
This example shows the use of nested grouped convolutions as well.
```
i  = randn(56,56,24,16)
a  = MeanPool((3,3), pad=(1,1), stride=(2,2))
b  = Chain(GroupedConvolutions(+, [Conv((1,1), 24=>64 , pad=(0,0)) for _ in 1:2]..., split=false),
           ChannelShuffle(2),
           DepthwiseConv((3,3), 64=>64, pad=(1,1), stride=(2,2)),
           GroupedConvolutions(+, [Conv((1,1), 64=>176, pad=(0,0)) for _ in 1:2]..., split=false))
g  = GroupedConvolutions((results...) -> cat(results..., dims=3), a, b, split=false)
o  = g(i)
```
"""
struct ChannelShuffle
  ngroups::Int

  function ChannelShuffle(ngroups::Int)
    ngroups > 1 || error("the number of groups (", ngroups, ") is not greater than 1")
    new(ngroups)
  end
end

@functor ChannelShuffle

function (shuffle::ChannelShuffle)(input)
  # get input size
  w::Int64, h::Int64, c::Int64, n::Int64 = size(input)
  # number of feature maps in input
  nmaps::Int64 = c
  # number of groups of the ChannelShuffle
  ngroups::Int64 = shuffle.ngroups
  # throw error if number of feature maps not divisible by number of paths
  mod(nmaps, ngroups*ngroups) == 0 || error("the number of feature maps in the input (", nmaps, ") is not divisible by the square of the number of groups of the ChannelShuffle (", ngroups*ngroups, ")")

  # number of maps per group
  nmaps_per_group::Int64 = div(nmaps, ngroups)

  # split up dimension of feature maps
  input = reshape(input, (w, h, nmaps_per_group, ngroups, n))
  # transpose the newly created dimensions, but not recursively
  input = permutedims(input, [1, 2, 4, 3, 5])
  # flatten the result back to the original dimensions
  reshape(input, (w, h, c, n))
end

function Base.show(io::IO, shuffle::ChannelShuffle)
  print(io, "ChannelShuffle(", shuffle.ngroups, ")")
end

"""
    ShuffledGroupedConvolutions(group, shuffle)
    ShuffledGroupedConvolutions(connection, paths..., split=false)

A wrapper around a subsequent `GroupedConvolutions` and `ChannelShuffle`.
Takes the number of paths in the grouped convolutions to be the number of groups in the channel shuffling operation.

Data should be stored in WHCN order (width, height, # channels, # batches).
In other words, a 100×100 RGB image would be a `100×100×3×1` array,
and a batch of 50 would be a `100×100×3×50` array.

The names of the variables are consistent accross all examples:
`i` stands for input,
`a` and `b` are `Chains`,
`g` represents a `GroupedConvolutions`,
`s` is a `SkipConnection`,
and `o` is the output.

Example A shows how to use shuffled grouped convolutions in practice for [ShuffleNet](https://arxiv.org/abs/1707.01083).
Batch Normalization and ReLU activations are left out for simplicity.

**Example A**: ShuffleNet v1 unit with stride=1.
(The numbers used in this example come from stage 2 and using 2 groups.)
```
i  = randn(28, 28, 200, 16)
c  = Chain(ShuffledGroupedConvolutions(+, [Conv((1,1), 200=>64, pad=(0,0)) for _ in 1:2]..., split=false),
          #ShuffledGroupedConvolutions(GroupedConvolutions(+, [Conv((1,1), 200=>64, pad=(0,0)) for _ in 1:2]..., split=false),
          #                            ChannelShuffle(2)),
           DepthwiseConv((3,3), 64=>64, pad=(1,1), stride=(1,1)),
           GroupedConvolutions(+, [Conv((1,1), 64=>200, pad=(0,0)) for _ in 1:2]..., split=false))
s  = SkipConnection(c, +)
o  = s(i)
```
"""
struct ShuffledGroupedConvolutions
  group::GroupedConvolutions
  shuffle::ChannelShuffle

  function ShuffledGroupedConvolutions(group::GroupedConvolutions, shuffle::ChannelShuffle)
    shuffle.ngroups == size(group.paths, 1) || error("the number of groups in the ChannelShuffle layer (", shuffle.ngroups, ") is not equal to the number of paths in the GroupedConvolutions (", size(group.paths, 1), ")")
    new(group, shuffle)
  end

  ShuffledGroupedConvolutions(connection, paths...; split::Bool=false) = new(GroupedConvolutions(connection, paths..., split=split), ChannelShuffle(size(paths, 1)))
end

@functor ShuffledGroupedConvolutions

function (shuffled::ShuffledGroupedConvolutions)(input)
  shuffled.shuffle(shuffled.group(input))
end

function Base.show(io::IO, shuffled::ShuffledGroupedConvolutions)
  print(io, shuffled.group, ", ", shuffled.shuffle)
end
