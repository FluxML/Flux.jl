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
type Chain
  layers::Vector{Any}
  Chain(xs...) = new([xs...])
end

@forward Chain.layers Base.getindex, Base.first, Base.last, Base.endof, Base.push!
@forward Chain.layers Base.start, Base.next, Base.done

children(c::Chain) = c.layers
mapchildren(f, c::Chain) = Chain(f.(c.layers)...)

(c::Chain)(x) = foldl((x, m) -> m(x), x, c.layers)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end

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
  σ::F
  W::S
  b::T
end

Dense(in::Integer, out::Integer, σ = identity; init = initn) =
  Dense(σ, param(init(out, in)), param(init(out)))

treelike(Dense)

function (a::Dense)(x)
  W, b, σ = a.W, a.b, a.σ
  σ.(W*x .+ b)
end

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end


"""
  Dropout(p; testmode=false)

A Dropout layer. If `testmode=false` mode sets input components `x[i]` to zero with
probability `p` and to `x[i]/(1-p)` with probability `(1-p)`.

In `testmode=true`it doesn't alter the input: `x == Dropout(p; mode=:eval)(x)`.
Change the mode with [`testmode!`](@ref).
"""
mutable struct Dropout{F}
  p::F
  testmode::Bool
end
Dropout(p::F; testmode::Bool=false) where {F} = Dropout{F}(p, testmode)

function (a::Dropout)(x)
  if a.testmode
    return x
  else
    if 0 < a.p < 1
      y = similar(x)
      rand!(y)
      q = 1 - a.p
      @inbounds for i=1:length(y)
        y[i] = y[i] > a.p ? 1 / q : 0
      end
      return y .* x
    elseif a.p == 0
      return x
    elseif a.p == 1
      return zeros(x)
    end
  end
end

"""
    testmode!(m, val=true)

Set model `m` in test mode if `val=true`, and in training mode otherwise.
This has an affect only if `m` contains [`Dropout`](@ref) or `BatchNorm` layers.
"""
testmode!(m, val::Bool=true) = prefor(x -> :testmode ∈ fieldnames(x) && (x.testmode = val), m)
