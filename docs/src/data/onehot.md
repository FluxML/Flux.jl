# One-Hot Encoding

It's common to encode categorical variables (like `true`, `false` or `cat`, `dog`) in "one-of-k" or ["one-hot"](https://en.wikipedia.org/wiki/One-hot) form. Flux provides the `onehot` function to make this easy.

```jldoctest onehot
julia> using Flux: onehot, onecold

julia> onehot(:b, [:a, :b, :c])
3-element Flux.OneHotVector{3,UInt32}:
 0
 1
 0

julia> onehot(:c, [:a, :b, :c])
3-element Flux.OneHotVector{3,UInt32}:
 0
 0
 1
```

The inverse is `onecold` (which can take a general probability distribution, as well as just booleans).

```jldoctest onehot
julia> onecold(ans, [:a, :b, :c])
:c

julia> onecold([true, false, false], [:a, :b, :c])
:a

julia> onecold([0.3, 0.2, 0.5], [:a, :b, :c])
:c
```

```@docs
Flux.onehot
Flux.onecold
```

## Batches

`onehotbatch` creates a batch (matrix) of one-hot vectors, and `onecold` treats matrices as batches.

```jldoctest onehot
julia> using Flux: onehotbatch

julia> onehotbatch([:b, :a, :b], [:a, :b, :c])
3×3 Flux.OneHotArray{UInt32,3,1,2,Vector{UInt32}}:
 0  1  0
 1  0  1
 0  0  0

julia> onecold(ans, [:a, :b, :c])	
3-element Vector{Symbol}:	
 :b	
 :a	
 :b   
```

Note that these operations returned `OneHotVector` and `OneHotMatrix` rather than `Array`s. `OneHotVector`s behave like normal vectors but avoid any unnecessary cost compared to using an integer index directly. For example, multiplying a matrix with a one-hot vector simply slices out the relevant row of the matrix under the hood.

```@docs
Flux.onehotbatch
```
