# Handling different image layouts

!!! tip
    The contents here are specific for images but similar reasoning can be applied to other data.

When using Flux to do image processing and computer vision tasks, there are often a few different
data layouts you'll encounter before passing your data into a network. For example, if you are trying to
represent 64 2-dimensional RGB images of size 32x32, you could possibly store them using any of the
following formats:

* Vector of arrays, where each array is an array of struct: `Vector{Matrix{RGB{Float32}}}`
* Vector of plain arrays: `Vector{Array{Float32, 3}}`
* Array of struct: `Array{RGB{Float32}, 3}`
* plain array: `Array{Float32, 4}`

Even within the plain array case, there are various data memory orders depending on which dimensions we put the
channel (C), height (H), width (H), and batch (N) information. Among all the possible combinations,
two of them are used most commonly in deep learning frameworks:

* WHCN: This is the Flux standard because Julia is column-major.
* NCHW: Pytorch standard because it uses row-major by default.
* CHWN: If we unroll the data of `Vector{Matrix{RGB{Float32}}}` without manipulate memory, we get
  this order.

Depending on what ecosystem you are referring to to, all of these layouts and memory orders are
useful. This page walks you through the transformaion between these data layouts. we will
also explain some of the confusions you might meet, or in other words, explain why we need all these
different layouts and why we can't just enforce one as a "standard."

We will start with a quick lookup table style so you can easily find out what you need. Then we
focus on the reasoning behind these layouts and their concrete usages in different ecosystems so you
will know how to choose the most appropriate layout for your specific applications and scenarios.

## The quick lookup table

It's probably easier and more accurate to use types rather than the long "vector of plain arrays"
description. Thus here we use 2D RGB images as an example, for other cases you will need to adjust it a
bit.

The main tools at hand are:

* `colorview`/`channelview` to splat and combine image channels. Requires [`ImageCore`](https://github.com/JuliaImages/ImageCore.jl) (or [`Images`](https://github.com/JuliaImages/Images.jl)).
* `permutedims` to change orders. `PermutedDimsArray` is the lazy version of it.
* `Flux.stack`/`Flux.unstack` are used to combining mutiple small arrays into one big array or vice
  versa. [`StackViews.StackView`](https://github.com/JuliaArrays/StackViews.jl) is lazy version of `Flux.stack`.

To reduce unnecessary memory allocations and get better transformation performance, we use lazy
versions in the table. Not every entry in this table is useful but I just leave them here as some
mental practice so that you can get better idea of how things work.

| From (order)                 | To (order)                    | Transformation
| ------------------------     | -----------------------       | -----------------------------------
| `Vector{Matrix{RGB{T}}}`(HW) | `Vector{Array{T,3}}`(CHW)     | `channelview.(X)`
| `Vector{Matrix{RGB{T}}}`(HW) | `Vector{Array{T,3}}`(HWC)     | `map(x->PermutedDimsArray(channelview(x), (2, 3, 1)), X)`
| `Vector{Matrix{RGB{T}}}`(HW) | `Array{RGB{T},3}`(WHN)        | `StackView(X)`
| `Vector{Matrix{RGB{T}}}`(HW) | `Array{RGB{T},3}`(NWH)        | `StackView(X, Val(1))`
| `Vector{Matrix{RGB{T}}}`(HW) | `Array{T,4}`(CHWN)            | `channelview(StackView(X))`
| `Vector{Matrix{RGB{T}}}`(HW) | `Array{T,4}`(HWCN)            | `PermutedDimsArray(channelview(StackView(X)), (2, 3, 1, 4))`
| `Vector{Array{T,3}}`(CHW)    | `Vector{Matrix{RGB{T}}}`(HW)  | `colorview.(RGB, X)`
| `Vector{Array{T,3}}`(CHW)    | `Vector{Array{T,3}}`(CHW)     | `PermutedDimsArray(X, (2, 3, 1))`
| `Vector{Array{T,3}}`(CHW)    | `Array{RGB{T},3}`(HWN)        | `StackView(colorview.(RGB, X))`
| `Vector{Array{T,3}}`(CHW)    | `Array{RGB{T},3}`(NHW)        | `StackView(colorview.(RGB, X), Val(1))`
| `Vector{Array{T,3}}`(CHW)    | `Array{T,4}`(CHWN)            | `StackView(X)`
| `Vector{Array{T,3}}`(CHW)    | `Array{T,4}`(HWCN)            | `PermutedDimsArray(StackView(X), (2, 3, 1, 4))`
| `Array{RGB{T},3}`(HWN)       | `Vector{Matrix{RGB{T}}}`(HW)  | `Flux.unstack(X)` (eager)
| `Array{RGB{T},3}`(HWN)       | `Vector{Array{T,3}}`(CHW)     | `Flux.unstack(channelview(X), 4)` (eager)
| `Array{RGB{T},3}`(HWN)       | `Vector{Array{T,3}}`(HWC)     | `Flux.unstack(PermutedDimsArray(channelview(X), (2, 3, 1, 4)), 4)` (eager)
| `Array{RGB{T},3}`(HWN)       | `Array{T,4}`(CHWN)            | `channelview(X)`
| `Array{RGB{T},3}`(HWN)       | `Array{T,4}`(HWCN)            | `PermutedDimsArray(channelview(X), (2, 3, 4, 1))`
| `Array{T,4}`(CHWN)           | `Array{T,4}`(HWCN)            | `PermutedDimsArray(X, (2, 3, 4, 1))`
| `Array{T,4}`(CHWN)           | `Array{RGB{T},3}`(HWN)        | `colorview(RGB, X)`
| `Array{T,4}`(CHWN)           | `Vector{Array{T,3}}`(CHW)     | `Flux.unstack(X, 4)` (eager)
| `Array{T,4}`(CHWN)           | `Vector{Matrix{RGB{T}}}`(HW)  | `Flux.unstack(colorview(RGB, X), 3)` (eager)
| `Array{T,4}`(HWCN)           | `Array{T,4}`(CHWN)            | `PermutedDimsArray(X, (3, 1, 2, 4))`
| `Array{T,4}`(HWCN)           | `Array{RGB{T},3}`(HWN)        | `colorview(RGB, PermutedDimsArray(X, (3, 1, 2, 4)))`
| `Array{T,4}`(HWCN)           | `Vector{Array{T,3}}`(CHW)     | `Flux.unstack(PermutedDimsArray(X, (3, 1, 2, 4)), 4)` (eager)
| `Array{T,4}`(HWCN)           | `Vector{Matrix{RGB{T}}}`(HW)  | `Flux.unstack(colorview(RGB, PermutedDimsArray(X, (3, 1, 2, 4))), 3)` (eager)

Addtional notes on this table:

* The listed transformations in the table are by no means the only possible version. The Julia
  ecosystem has very generic array operations support, so you're invited to try other packages and
  see if that fits your usage better.
* Height (H) and width (W) dimensions are often used interchangeably so we don't differentiate them
  here. If that matters to you, then you should be able to add or modify the `permutedims`/`PermutedDimsArray`
  arguments.
* If you're processing GPU arrays where `getindex` performance is extremely slow, you should
  probably use the eager vector versions or insert some `collect`. Generally, a simple strategy is to
  do all the permutations on CPU until you upload them to GPU.
* For performance consideration, the code here doesn't necessarily return the exact type declared in
  the table. For instance, `channelview` outputs `ReinterpretArray` and not `Array`. In most cases
  this doesn't matter at all, but if it matters, calling `collect` or similar functions could
  generate a dense representation of it.
* Methods in this table might not be AD-ready.

## Why there isn't one single function that just works for every different layout transformations?

It is theoretically possible but the entire ecosystem (Flux, JuliaImages, CUDA.jl and others) is not
yet ready. The main reason is that in Julia we want to write generic codes that works for all
different use cases. Hardcoding the meaning of plain numerical array `Array{<:Number, 4}` in any
format, say WHCN in Flux, would introduce unavoidable ambiguities. To make this actual work in practice
we need to give more type and dimension information to do dispatch (e.g., `RGB`) but this requires a
lot of fundamental supporting work from different Julia communities.

So why does Flux use WHCN format at present? This is mainly because Flux is backed by the C/C++
libraries (e.g., NVIDIA CuDNN) to do the computation for GPU arrays, which hardcode NCHW format (C/C++ is row-major order); we
will explain this further later. When Flux is smart enough to know how to unroll the data into raw numerical array and feed
to different backends, it is then possible that things just work for any generic array layout.

## What data layout should I use

Let's first start with a statement: there is no best data layout that suits every use case; whenever
you choose a data layout, you are making tradeoffs between usability and performance. For
better usability we write code at individual element-wise level, and for better performance we
emphasize the data locality for the specific computation pattern.

### Represent an image

The first layout choice that you might notice is Array of Struct (AOS) and Struct of Array (AOS).
We need to differentiate these because they are stored quite differently in memory: AOS in flat
memory representation is `RGBRGB...RGB` while SOA in flat memory representation is
`RR...RGG...GBB...B`.

When you store an image as `Array{<:RGB}`, you are storing it in AOS layout:

```julia
using ImageCore
using TestImages
using BenchmarkTools

rgb = [RGB(0.1, 0.2, 0.3), RGB(0.4, 0.5, 0.6), RGB(0.7, 0.8, 0.9)] # Vector{RGB{N0f8}}

# channelview opens the colorant struct without manipulating its memory
channelview(rgb)[:] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # true
```

and when you're storing a RGB image as `Array{Float32, 3}`(HWC), you're indeed storing it in SOA layout:

```julia
rgb_soa = [0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9]
red_channel = rgb_soa[:, 1]
```

Due to the memory layout difference of AOS and SOA layout, the first and the most important
consequence is: AOS layout works best if the underlying function is defined per pixel because it
benifits from the data locality (e.g., CPU cache).

```julia
function colorjitter(img)
    bias = 0.1rand(eltype(img))
    return @. clamp01(img + bias)
end
function colorjitter_cv(img)
    out = similar(img)
    @inbounds for i in axes(img, 3)
        bias = 0.1rand(eltype(img))
        channel = @view img[:, :, i]
        @. out[:, :, i] = clamp01(channel + bias)
    end
    return out
end

# array of struct layout
img = testimage("lighthouse"); # size (512, 768)
@btime colorjitter($img); # 1.035 ms (2 allocations: 9.00 MiB)

# struct of array layout
img = permutedims(channelview(testimage("lighthouse")), (2, 3, 1)) # size (512, 768, 3)
@btime colorjitter_cv($img); # 2.215 ms (2 allocations: 1.13 MiB)
```

Most kernel operations in JuliaImages are defined per-pixel, thus using AOS layout has better
performance. For a similar reason, SOA layout works best if the underlying function is defined per
channel. Because deep learning uses convolutional neural networks (CNNs) for image processing and
computer vision tasks, and because CNN is a set of filter operations defined per channel, CNN
backends (e.g., CuDNN) uses SOA layout to squeeze every last drop of performance out of the GPUs.

!!! note
    Modern hardware and compiler tricks are more complicated and sophisticated such that this
    SOA-AOS performance gap may not be as big as you can expect if enough engineer efforts were paid.

Except for the performance part, there are two other valid reasons on why JuliaImages uses the
array of struct layout:

* Julia's broadcasting is so powerful that you can only benefit from its simplicity if you commit to
  the "every element of array is a computational scalar" assumption, and define colorant structs
  such as `RGB` and store RGB image as `Array{RGB}` is a natural step towards this.
* JuliaImages is designed to process generic images (not just 2D RGB images), and using plain
  `Array{Float32, 3}` to store data will cause ambiguities.

### Represent a batch

`Vector{Array{Float32, 3}}` (WHC) and `Array{Float32, 4}` (WHCN) are two typical ways to store RGB
images in SOA layout. What's the benefits of one over the other since they have the same memory
layout?

In Julia there isn't much difference because we have much better high-order function supports.
Assume that we have defined a _performant_ function `fn` to process one single image `Array{Float32, 3}`, it is
clear that storing data in `Vector{Array}` makes things easier to write without losing performance.

```julia
# If it is Vector{Array{Float32, 3}}
map(fn, batch) # much simpler!

# If it is Array{Float32, 4}
out = similar(batch)
for i in axes(batch, 4)
    out[:, :, :, i] .= fn(@view(batch[:, :, :, i]))
end
```

But in Python, things will be quite different because both for-loop and `map` will introduce
significant overhead due to the dynamic nature of Python. Thus the solution is to vectorize[1] the
data by providing a bigger function `fn_vec` to handle a batch of images intead of one single image
so that `fn_vec(batch)` just works. Any network layer in Flux can be seen as `fn_vec`. The reason
`Array{Float32, 4}` is used by the C/C++ backends is because it has the simplest memory structure; a
big memory block with plain old data.

Then the conclusion: if we can build everything using pure Julia, then `Vector{Array{Float32, 3}}`
layout is the simplest solution. But the reality is that we don't have a Julia version of CuDNN that
has competitive performance. Thus for quite a long time in the future, we have to live with the
ambiguous `Array{Float32, 4}` layout.

> Disclamer: I know there are many tricks in CUDA kernel programming to reduce the data transferring
> waiting time and to increase the concurrency, but I'm not a CUDA expert and I don't know how those
> tricks apply to `Vector{Array{Float32, 3}}` and `Array{Float32, 4}` layouts.

## References

* [1] For broadcasting and vectorization, please read [More Dots: Syntactic Loop Fusion in
  Julia](https://julialang.org/blog/2017/01/moredots/)
