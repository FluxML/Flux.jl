


"""
    Join(parallel_layers...)

Create a new `Join` layer, which deploys multiple inputs to multiple paths and 
concatenates the results to a single output vector.

The input `x` must be a tuple of length `parallel_layers`. 
The out `y` will be a vector.

# Example
```
julia> using flux

julia> using CUDA

julia> model = Chain(
  Join(
    Chain(
      Dense(1, 5),
      Dense(5, 1)
    ),
    Dense(1, 2),
    Dense(1, 1),
  ),
  Dense(4, 1)
) |> gpu

julia> model(cu(rand(1), rand(1), rand(1)))
Float32[0.32]
```
"""
struct Join{F}
  fs::F
end

function Join(fs ...)
  Join(fs)
end

@functor Join

function (w::Join)(t::Tuple)
  # may have performance issues ... some julia/flux pro should maybe rework this
  @assert length(w.fs) == length(t)
  vcat([w.fs[i](t[i]) for i in 1:length(w.fs)]...)
end

Base.getindex(c::Join, i::AbstractArray) = Join(c.fs[i]...)

function Base.show(io::IO, j::Join)
  print(io, "Join(", length(fs), ", ")
  join(io, j.fs, ", ")
  print(io, ")")
end # todo test


"""
    Split(parallel_layers...)

Create a new `Split` layer, which deploys the input array to multiple paths and 
outputs the results as a tuple with multiple vector.

The input `x` must be a common input format. 
The out `y` will be a tuple with n vectors.

# Example
```
julia> using flux

julia> using CUDA

julia> model = Chain(
  Dense(1, 1),
  Split(
    Dense(1, 1),
    Dense(1, 1),
    Dense(1, 1)
  )
) |> gpu

julia> model(cu(rand(1)))
(Float32[0.12], Float32[0.52], Float32[0.23])
```
"""
struct Split{F}
  fs::F
end

function Split(fs...)
  Split(fs)
end

Flux.@functor Split

function (w::Split)(x::AbstractArray)
  # may have performance issues ... some julia/flux pro should maybe rework this
  tuple([w.fs[i](x) for i in 1:length(w.fs)])
end

Base.getindex(c::Split, i::AbstractArray) = Split(c.fs[i]...)

function Base.show(io::IO, j::Split)
  print(io, "Split(", length(fs), ", ")
  join(io, j.fs, ", ")
  print(io, ")")
end # todo test


"""
    Parallel(parallel_layers...)

Create a new `Parallel` layer, which deploys the input array into multiple paths 
and concatenates the results to a single output vector.

The input `x` must be a common input format. 
The out `y` will be vector.

# Example
```
julia> using flux

julia> using CUDA

julia> model = Chain(
  Dense(1, 1),
  Parallel(
    Dense(1, 1),
    Dense(1, 3),
    Chain(
      Dense(1, 5),
      Dense(5, 2),
    )
  ),
  Dense(6, 1)
) |> gpu

julia> model(cu((rand(1))))
Float32[0.27]
```
"""
struct Parallel{F}
  fs::F
end

function Parallel(fs...)
  Parallel(fs)
end

Flux.@functor Parallel

function (w::Parallel)(x::AbstractArray)
  # may have performance issues ... some julia/flux pro should maybe rework this
  vcat([w.fs[i](x) for i in 1:length(w.fs)]...)
end

Base.getindex(c::Parallel, i::AbstractArray) = Parallel(c.fs[i]...)

function Base.show(io::IO, j::Parallel)
  print(io, "Parallel(", length(fs), ", ")
  join(io, j.fs, ", ")
  print(io, ")")
end # todo test







