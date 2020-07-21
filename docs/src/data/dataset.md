# Custom Dataset

If you can't represent your dataset as a single `Array`,
for example it does not fit into memory,
you can define a custom `struct` that would load items on demand.
This structure needs to subtype `AbstractArray` and implement following methods:

- `Base.getindex(::MyDataset, i::Int)` -- how to load one item from dataset;
- `Base.getindex(::MyDataset, ids::Union{Array, UnitRange})` -- how to load mini-batch.
  This is needed, if your dataset at index `i` returns complex structures (e.g. multidimensional arrays, tuples of arrays, etc.).
  Otherwise this will be handled automatically;
- `Base.length(::MyDataset)` -- length of dataset;
- `Base.size(::MyDataset)` -- last dimension should equal to the length of dataset.

Below is an example of how you might implement custom dataset that consists of
images and for each image we also return target value.

Define structure:

```julia
struct MyDataset <: AbstractArray{Float32, 1}
    frame_template::Formatting.FormatExpr  # Template path for loading frames
    targets::AbstractArray                 # Array of targets for each image
end
```

Define how to load one item for a given index `i`:

```julia
function Base.getindex(d::MyDataset, i::Int)
    path = Formatting.format(d.frame_template, i)
    image = path |> FileIO.load |> Images.channelview .|> Float32
    image, d.targets[[i]]
end
```

How to load mini-batch given array of indices or range:

```julia
function Base.getindex(d::MyDataset, ids::Union{Array, UnitRange})
    x, y = d[ids[1]]
    xs_last_dim = ntuple(i -> Colon(), ndims(x))
    ys_last_dim = ntuple(i -> Colon(), ndims(y))

    xs = Array{eltype(x)}(undef, size(x)..., length(ids))
    ys = Array{eltype(y)}(undef, size(y)..., length(ids))

    xs[xs_last_dim..., 1] .= x
    ys[ys_last_dim..., 1] .= y

    for (i, id) in enumerate(ids[2:end])
        x, y = d[id]
        xs[xs_last_dim..., i + 1] .= x
        ys[ys_last_dim..., i + 1] .= y
    end
    xs, ys
end
```

And finally some helper functions:

```julia
Base.IndexStyle(::Type{MyDataset}) = IndexLinear()
Base.size(d::MyDataset) = (length(d.targets),)
Base.length(d::MyDataset) = length(d.targets)
```

After that you can use `MyDataset` with `Dataloader`:

```julia
dataset = MyDataset(
    Formatting.FormatExpr(raw"frames\frame-{:d}.jpg"),
    randn(Float32, 100),
)
loader = Flux.Data.DataLoader(dataset, batchsize=4, shuffle=true)
for (i, (images, targets)) in enumerate(loader)
    println("$i: $(size(images)) $(size(targets))")
end
```
