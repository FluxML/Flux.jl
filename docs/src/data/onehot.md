# One-Hot Encoding with OneHotArrays.jl

It's common to encode categorical variables (like `true`, `false` or `cat`, `dog`) in "one-of-k" or ["one-hot"](https://en.wikipedia.org/wiki/One-hot) form. [OneHotArrays.jl](https://github.com/FluxML/OneHotArrays.jl) provides the `onehot` function to make this easy.

```jldoctest onehot
julia> using OneHotArrays

julia> onehot(:b, [:a, :b, :c])
3-element OneHotVector(::UInt32) with eltype Bool:
 ⋅
 1
 ⋅

julia> onehot(:c, [:a, :b, :c])
3-element OneHotVector(::UInt32) with eltype Bool:
 ⋅
 ⋅
 1
```

There is also a `onecold` function, which is an inverse of `onehot`. It can also be given an array of numbers instead of booleans, in which case it performs an `argmax`-like operation, returning the label with the highest corresponding weight.

```jldoctest onehot
julia> onecold(ans, [:a, :b, :c])
:c

julia> onecold([true, false, false], [:a, :b, :c])
:a

julia> onecold([0.3, 0.2, 0.5], [:a, :b, :c])
:c
```

For multiple samples at once, `onehotbatch` creates a batch (matrix) of one-hot vectors, and `onecold` treats matrices as batches.

```jldoctest onehot
julia> using OneHotArrays

julia> onehotbatch([:b, :a, :b], [:a, :b, :c])
3×3 OneHotMatrix(::Vector{UInt32}) with eltype Bool:
 ⋅  1  ⋅
 1  ⋅  1
 ⋅  ⋅  ⋅

julia> onecold(ans, [:a, :b, :c])
3-element Vector{Symbol}:
 :b
 :a
 :b
```

Note that these operations returned `OneHotVector` and `OneHotMatrix` rather than `Array`s. `OneHotVector`s behave like normal vectors but avoid any unnecessary cost compared to using an integer index directly. For example, multiplying a matrix with a one-hot vector simply slices out the relevant row of the matrix under the hood.

### Function listing

```@docs
OneHotArrays.onehot
OneHotArrays.onecold
OneHotArrays.onehotbatch
OneHotArrays.OneHotArray
OneHotArrays.OneHotVector
OneHotArrays.OneHotMatrix
```
