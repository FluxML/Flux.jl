# One-Hot Encoding

It's common to encode categorical variables (like `true`, `false` or `cat`, `dog`) in "one-of-k" or ["one-hot"](https://en.wikipedia.org/wiki/One-hot) form. Flux provides the `onehot` function to make this easy.

```
julia> using Flux: onehot, onecold

julia> onehot(:b, [:a, :b, :c])
3-element Flux.OneHotVector:
 false
  true
 false

julia> onehot(:c, [:a, :b, :c])
3-element Flux.OneHotVector:
 false
 false
  true
```

The inverse is `onecold` (which can take a general probability distribution, as well as just booleans).

```julia
julia> onecold(ans, [:a, :b, :c])
:c

julia> onecold([true, false, false], [:a, :b, :c])
:a

julia> onecold([0.3, 0.2, 0.5], [:a, :b, :c])
:c
```

## Batches

`onehotbatch` creates a batch (matrix) of one-hot vectors, and `onecold` treats matrices as batches.

```julia
julia> using Flux: onehotbatch

julia> onehotbatch([:b, :a, :b], [:a, :b, :c])
3×3 Flux.OneHotMatrix:
 false   true  false
  true  false   true
 false  false  false

julia> onecold(ans, [:a, :b, :c])
3-element Array{Symbol,1}:
  :b
  :a
  :b
```

Note that these operations returned `OneHotVector` and `OneHotMatrix` rather than `Array`s. `OneHotVector`s behave like normal vectors but avoid any unnecessary cost compared to using an integer index directly. For example, multiplying a matrix with a one-hot vector simply slices out the relevant row of the matrix under the hood.
