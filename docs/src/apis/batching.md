# Batching

## Basics

Existing machine learning frameworks and libraries represent batching, and other properties of data, only implicitly. Your machine learning data is a large `N`-dimensional array, which may have a shape like:

```julia
100 × 50 × 256 × 256
```

Typically, this might represent that you have (say) a batch of 100 samples, where each sample is a 50-long sequence of 256×256 images. This is great for performance, but array operations often become much more cumbersome as a result. Especially if you manipulate dimensions at runtime as an optimisation, debugging models can become extremely fiddly, with a proliferation of `X × Y × Z` arrays and no information about where they came from.

Flux introduces a new approach where the batch dimension is represented explicitly as part of the data. For example:

```julia
julia> xs = Batch([[1,2,3], [4,5,6]])
2-element Batch of Vector{Int64}:
 [1,2,3]
 [4,5,6]
```

Batches are represented the way we *think* about them; as a list of data points. We can do all the usual array operations with them, including getting the first with `xs[1]`, iterating over them and so on. The trick is that under the hood, the data is batched into a single array:

```julia
julia> rawbatch(xs)
2×3 Array{Int64,2}:
 1  2  3
 4  5  6
```

When we put a `Batch` object into a model, the model is ultimately working with a single array, which means there's no performance overhead and we get the full benefit of standard batching.

Turning a set of vectors into a matrix is fairly easy anyway, so what's the big deal? Well, it gets more interesting as we start working with more complex data. Say we were working with 4×4 images:

```julia
julia> xs = Batch([[1 2; 3 4], [5 6; 7 8]])
2-element Flux.Batch of Array{Int64,2}:
 [1 2; 3 4]
 [5 6; 7 8]
```

The raw batch array is much messier, and harder to recognise:

```julia
julia> rawbatch(xs)
2×2×2 Array{Int64,3}:
[:, :, 1] =
 1  3
 5  7

[:, :, 2] =
 2  4
 6  8
```

Furthermore, because the batches acts like a list of arrays, we can use simple and familiar operations on it:

```julia
julia> map(flatten, xs)
2-element Array{Array{Int64,1},1}:
 [1,3,2,4]
 [5,7,6,8]
```

`flatten` is simple enough over a single data point, but flattening a batched data set is more complex and you end up needing arcane array operations like `mapslices`. A `Batch` can just handle this for you for free, and more importantly it ensures that your operations are *correct* – that you haven't mixed up your batch and data dimensions, or used the wrong array op, and so on.

## Sequences and Nesting

As well as `Batch`, there's a structure called `Seq` which behaves very similarly. Let's say we have two one-hot encoded DNA sequences:

```julia
julia> x1 = Seq([[0,1,0,0], [1,0,0,0], [0,0,0,1]]) # [A, T, C, G]
julia> x2 = Seq([[0,0,1,0], [0,0,0,1], [0,0,1,0]])

julia> rawbatch(x1)
3×4 Array{Int64,2}:
 0  1  0  0
 1  0  0  0
 0  0  0  1
```

This is identical to `Batch` so far; but where it gets interesting is that you can actually nest these types:

```julia
julia> xs = Batch([x1, x2])
2-element Batch of Seq of Vector{Int64}:
 [[0,1,0,0],[1,0,0,0],[0,0,0,1]]
 [[0,0,1,0],[0,0,0,1],[0,0,1,0]]
```

Again, this represents itself intuitively as a list-of-lists-of-lists, but `rawbatch` shows that the real underlying value is an `Array{Int64,3}` of shape `2×3×4`.

## Future Work

The design of batching is still a fairly early work in progress, though it's used in a few places in the system. For example, all Flux models expect to be given `Batch` objects which are unwrapped into raw arrays for the computation. Models will convert their arguments if necessary, so it's convenient to call a model with a single data point like `f([1,2,3])`.

Right now, the `Batch` or `Seq` types always stack along the left-most dimension. In future, this will be customisable, and Flux will provide implementations of common functions that are generic across the batch dimension. This brings the following benefits:

* Code can be written in a batch-agnostic way or be generic across batching strategies.
* Batching and optimisations, like switching batch dimensions, can be expressed by the programmer with compiler support; fewer code changes are required and optimisations are guaranteed not to break the model.
* This also opens the door for more automatic optimisations, e.g. having the compiler explore the search base of possible batching combinations.

Here's a more detailed illustration of how it might look for code to be "generic across batching". Take for example a weight matrix `W` times a vector `x`, as used in a logistic regression or a simple neural network:

```julia
   W    *   x  =>   y
(10×28) * (28) => (10)
```

If we want to work with a batch of 50 `x`s, one option is to stack the data into a matrix of size `28 × 50`.

```julia
   W    *    x    =>    y
(10×28) * (28×50) => (10×50)
```

This works, but we may find that it's slow or doesn't fit well with the rest of the model, which batches on the first dimension. For that reason we may instead want to put the data in a `50 × 28` matrix and alter the code as follows:

```julia
   x    *    W'   =>    y
(50×28) * (28×10) => (50×10)
```

to make the shapes work out. This code change is not ideal; in more complex cases it can become fiddly and error-prone, and it means that the code is less reusable, tied to a particular implementation strategy.

There's an alternative. We keep the same code, but represent the batched `x`s as either a `Batch{Vector,1}` or a `Batch{Vector,2}`, depending on how the data is stacked. Then we can simply overload `*` as follows:

```julia
*(W::Matrix, x::Batch{Vector,1}) = x * W'
*(W::Matrix, x::Batch{Vector,2}) = W * x
```

This means that we can always write `W*x`, and the code is reusable in a larger network regardless of the overall batching approach. Moreover, Julia's type system ensures there's no runtime cost to doing this, and we can compile the code appropriately for backends like TensorFlow as well.
