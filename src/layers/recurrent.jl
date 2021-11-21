
gate(h, n) = (1:h) .+ h*(n-1)
gate(x::AbstractVector, h, n) = @view x[gate(h,n)]
gate(x::AbstractMatrix, h, n) = x[gate(h,n),:]

# Stateful recurrence

"""
    Recur(cell)

`Recur` takes a recurrent cell and makes it stateful, managing the hidden state
in the background. `cell` should be a model of the form:

    h, y = cell(h, x...)

For example, here's a recurrent network that keeps a running total of its inputs:

```julia
accum(h, x) = (h + x, x)
rnn = Flux.Recur(accum, 0)
rnn(2)      # 2
rnn(3)      # 3
rnn.state   # 5
rnn.(1:10)  # apply to a sequence
rnn.state   # 60
```

Folding over a 3d Array of dimensions `(features, batch, time)` is also supported:

```julia
accum(h, x) = (h .+ x, x)
rnn = Flux.Recur(accum, zeros(Int, 1, 1))
rnn([2])                    # 2
rnn([3])                    # 3
rnn.state                   # 5
rnn(reshape(1:10, 1, 1, :)) # apply to a sequence of (features, batch, time)
rnn.state                   # 60
```

"""
mutable struct Recur{T,S}
  cell::T
  state::S
end

function (m::Recur)(x)
  m.state, y = m.cell(m.state, x)
  return y
end

@functor Recur
trainable(a::Recur) = (a.cell,)

Base.show(io::IO, m::Recur) = print(io, "Recur(", m.cell, ")")

"""
    reset!(rnn)

Reset the hidden state of a recurrent layer back to its original value.

Assuming you have a `Recur` layer `rnn`, this is roughly equivalent to:
```julia
rnn.state = hidden(rnn.cell)
```
"""
reset!(m::Recur) = (m.state = m.cell.state0)
reset!(m) = foreach(reset!, functor(m)[1])


# TODO remove in v0.13
function Base.getproperty(m::Recur, sym::Symbol)
  if sym === :init
    Zygote.ignore() do
      @warn "Recur field :init has been deprecated. To access initial state weights, use m::Recur.cell.state0 instead."
    end
    return getfield(m.cell, :state0)
  else
    return getfield(m, sym)
  end
end

flip(f, xs) = reverse(f.(reverse(xs)))

function (m::Recur)(x::AbstractArray{T, 3}) where T
  h = [m(view(x, :, :, i)) for i in 1:size(x, 3)]
  sze = size(h[1])
  reshape(reduce(hcat, h), sze[1], sze[2], length(h))
end

# Vanilla RNN

struct RNNCell{F,A,V,S}
  σ::F
  Wi::A
  Wh::A
  b::V
  state0::S
end

RNNCell(in::Integer, out::Integer, σ=tanh; init=Flux.glorot_uniform, initb=zeros32, init_state=zeros32) = 
  RNNCell(σ, init(out, in), init(out, out), initb(out), init_state(out,1))

function (m::RNNCell{F,A,V,<:AbstractMatrix{T}})(h, x::Union{AbstractVecOrMat{T},OneHotArray}) where {F,A,V,T}
  σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
  h = σ.(Wi*x .+ Wh*h .+ b)
  sz = size(x)
  return h, reshape(h, :, sz[2:end]...)
end

@functor RNNCell

function Base.show(io::IO, l::RNNCell)
  print(io, "RNNCell(", size(l.Wi, 2), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    RNN(in::Integer, out::Integer, σ = tanh)

The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.

The parameters `in` and `out` describe the size of the feature vectors passed as input and as output. That is, it accepts a vector of length `in` or a batch of vectors represented as a `in x B` matrix and outputs a vector of length `out` or a batch of vectors of size `out x B`.

This constructor is syntactic sugar for `Recur(RNNCell(a...))`, and so RNNs are stateful. Note that the state shape can change depending on the inputs, and so it is good to `reset!` the model between inference calls if the batch size changes. See the examples below.

# Examples
```jldoctest
julia> r = RNN(3, 5)
Recur(
  RNNCell(3, 5, tanh),                  # 50 parameters
)         # Total: 4 trainable arrays, 50 parameters,
          # plus 1 non-trainable, 5 parameters, summarysize 432 bytes.

julia> r(rand(Float32, 3)) |> size
(5,)

julia> Flux.reset!(r);

julia> r(rand(Float32, 3, 64)) |> size
(5, 64)

julia> Flux.reset!(r);

julia> r(rand(Float32, 3))
5-element Vector{Float32}:
 -0.37216917
 -0.14777198
  0.2281275
  0.32866752
 -0.6388411

# A demonstration of not using `reset!` when the batch size changes.
julia> r = RNN(3, 5)
Recur(
  RNNCell(3, 5, tanh),                  # 50 parameters
)         # Total: 4 trainable arrays, 50 parameters,
          # plus 1 non-trainable, 5 parameters, summarysize 432 bytes.

julia> r.state |> size
(5, 1)

julia> r(rand(Float32, 3))
5-element Vector{Float32}:
  0.3572996
 -0.041238427
  0.19673917
 -0.36114445
 -0.0023919558

julia> r.state |> size
(5, 1)

julia> r(rand(Float32, 3, 10)) # batch size of 10
5×10 Matrix{Float32}:
  0.50832    0.409913    0.392907    0.838393   0.297105    0.432568    0.439304   0.677793   0.690217   0.78335
 -0.36385   -0.271328   -0.405521   -0.443976  -0.279546   -0.171614   -0.328029  -0.551147  -0.272327  -0.336688
  0.272917  -0.0155508   0.0995184   0.580889   0.0502855   0.0375683   0.163693   0.39545    0.294581   0.461731
 -0.353226  -0.924237   -0.816582   -0.694016  -0.530896   -0.783385   -0.584767  -0.854036  -0.832923  -0.730812
  0.418002   0.657771    0.673267    0.388967   0.483295    0.444058    0.490792   0.707697   0.435467   0.350789

julia> r.state |> size # state shape has changed
(5, 10)

julia> r(rand(Float32, 3)) # outputs a length 5*10 = 50 vector.
50-element Vector{Float32}:
  0.8532559
 -0.5693587
  0.49786803
  ⋮
 -0.7722325
  0.46099305
```

"""
RNN(a...; ka...) = Recur(RNNCell(a...; ka...))
Recur(m::RNNCell) = Recur(m, m.state0)

# TODO remove in v0.13
function Base.getproperty(m::RNNCell, sym::Symbol)
  if sym === :h
    Zygote.ignore() do
      @warn "RNNCell field :h has been deprecated. Use m::RNNCell.state0 instead."
    end
    return getfield(m, :state0)
  else
    return getfield(m, sym)
  end
end

# LSTM

struct LSTMCell{A,V,S}
  Wi::A
  Wh::A
  b::V
  state0::S
end

function LSTMCell(in::Integer, out::Integer;
                  init = glorot_uniform,
                  initb = zeros32,
                  init_state = zeros32)
  cell = LSTMCell(init(out * 4, in), init(out * 4, out), initb(out * 4), (init_state(out,1), init_state(out,1)))
  cell.b[gate(out, 2)] .= 1
  return cell
end

function (m::LSTMCell{A,V,<:NTuple{2,AbstractMatrix{T}}})((h, c), x::Union{AbstractVecOrMat{T},OneHotArray}) where {A,V,T}
  b, o = m.b, size(h, 1)
  g = m.Wi*x .+ m.Wh*h .+ b
  input = σ.(gate(g, o, 1))
  forget = σ.(gate(g, o, 2))
  cell = tanh.(gate(g, o, 3))
  output = σ.(gate(g, o, 4))
  c = forget .* c .+ input .* cell
  h′ = output .* tanh.(c)
  sz = size(x)
  return (h′, c), reshape(h′, :, sz[2:end]...)
end

@functor LSTMCell

Base.show(io::IO, l::LSTMCell) =
  print(io, "LSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4, ")")

"""
    LSTM(in::Integer, out::Integer)

[Long Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
recurrent layer. Behaves like an RNN but generally exhibits a longer memory span over sequences.

The parameters `in` and `out` describe the size of the feature vectors passed as input and as output. That is, it accepts a vector of length `in` or a batch of vectors represented as a `in x B` matrix and outputs a vector of length `out` or a batch of vectors of size `out x B`.

This constructor is syntactic sugar for `Recur(LSTMCell(a...))`, and so LSTMs are stateful. Note that the state shape can change depending on the inputs, and so it is good to `reset!` the model between inference calls if the batch size changes. See the examples below.

See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.

  # Examples
  ```jldoctest
  julia> l = LSTM(3, 5)
  Recur(
    LSTMCell(3, 5),                       # 190 parameters
  )         # Total: 5 trainable arrays, 190 parameters,
            # plus 2 non-trainable, 10 parameters, summarysize 1.062 KiB.
  
  julia> l(rand(Float32, 3)) |> size
  (5,)
  
  julia> Flux.reset!(l);
  
  julia> l(rand(Float32, 3, 10)) |> size # batch size of 10
  (5, 10)
  
  julia> Flux.reset!(l);
  
  julia> l(rand(Float32, 3))
  5-element Vector{Float32}:
   -0.025144277
    0.03836835
    0.13517386
   -0.028824253
   -0.057356793
  
  # A demonstration of not using `reset!` when the batch size changes.
  julia> l = LSTM(3, 5)
  Recur(
    LSTMCell(3, 5),                       # 190 parameters
  )         # Total: 5 trainable arrays, 190 parameters,
            # plus 2 non-trainable, 10 parameters, summarysize 1.062 KiB.
  
  julia> size.(l.state)
  ((5, 1), (5, 1))
  
  julia> l(rand(Float32, 3))
  5-element Vector{Float32}:
   0.038496178
   0.047853474
   0.025309514
   0.0934924
   0.05440048
  
  julia> l(rand(Float32, 3, 10)) # batch size of 10
  5×10 Matrix{Float32}:
   0.169775   -0.0268295  0.0985312  0.0335569  0.023051   0.146001   0.0494771  0.12347    0.148342    0.00534695
   0.0784295   0.130255   0.0326518  0.0495609  0.108738   0.10251    0.0519795  0.0673814  0.0804598   0.135432
   0.109187   -0.0267218  0.0772971  0.0200508  0.0108066  0.0921862  0.0346887  0.0831271  0.0978057  -0.00210143
   0.0827624   0.163729   0.10911    0.134769   0.120407   0.0757773  0.0894074  0.130243   0.0895137   0.133424
   0.060574    0.127245   0.0145216  0.0635873  0.108584   0.0954128  0.0529619  0.0665022  0.0689427   0.127494
  
  julia> size.(l.state) # state shape has changed
  ((5, 10), (5, 10))
  
  julia> l(rand(Float32, 3)) # outputs a length 5*10 = 50 vector.
  50-element Vector{Float32}:
    0.07209678
    0.1450204
    ⋮
    0.14622498
    0.15595339
```
"""
LSTM(a...; ka...) = Recur(LSTMCell(a...; ka...))
Recur(m::LSTMCell) = Recur(m, m.state0)

# TODO remove in v0.13
function Base.getproperty(m::LSTMCell, sym::Symbol)
  if sym === :h
    Zygote.ignore() do
      @warn "LSTMCell field :h has been deprecated. Use m::LSTMCell.state0[1] instead."
    end
    return getfield(m, :state0)[1]
  elseif sym === :c
    Zygote.ignore() do
      @warn "LSTMCell field :c has been deprecated. Use m::LSTMCell.state0[2] instead."
    end  
    return getfield(m, :state0)[2]
  else
    return getfield(m, sym)
  end
end

# GRU

function _gru_output(Wi, Wh, b, x, h)
  o = size(h, 1)
  gx, gh = Wi*x, Wh*h
  r = σ.(gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
  z = σ.(gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))

  return gx, gh, r, z
end

struct GRUCell{A,V,S}
  Wi::A
  Wh::A
  b::V
  state0::S
end

GRUCell(in, out; init = glorot_uniform, initb = zeros32, init_state = zeros32) =
  GRUCell(init(out * 3, in), init(out * 3, out), initb(out * 3), init_state(out,1))

function (m::GRUCell{A,V,<:AbstractMatrix{T}})(h, x::Union{AbstractVecOrMat{T},OneHotArray}) where {A,V,T}
  b, o = m.b, size(h, 1)
  gx, gh, r, z = _gru_output(m.Wi, m.Wh, b, x, h)
  h̃ = tanh.(gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
  h′ = (1 .- z) .* h̃ .+ z .* h
  sz = size(x)
  return h′, reshape(h′, :, sz[2:end]...)
end

@functor GRUCell

Base.show(io::IO, l::GRUCell) =
  print(io, "GRUCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ")")

"""
    GRU(in::Integer, out::Integer)

[Gated Recurrent Unit](https://arxiv.org/abs/1406.1078v1) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences. This implements
the variant proposed in v1 of the referenced paper.

The parameters `in` and `out` describe the size of the feature vectors passed as input and as output. That is, it accepts a vector of length `in` or a batch of vectors represented as a `in x B` matrix and outputs a vector of length `out` or a batch of vectors of size `out x B`.

This constructor is syntactic sugar for `Recur(GRUCell(a...))`, and so GRUs are stateful. Note that the state shape can change depending on the inputs, and so it is good to `reset!` the model between inference calls if the batch size changes. See the examples below.

See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.

# Examples
```jldoctest
julia> g = GRU(3, 5)
Recur(
  GRUCell(3, 5),                        # 140 parameters
)         # Total: 4 trainable arrays, 140 parameters,
          # plus 1 non-trainable, 5 parameters, summarysize 792 bytes.

julia> g(rand(Float32, 3)) |> size
(5,)

julia> Flux.reset!(g);

julia> g(rand(Float32, 3, 10)) |> size # batch size of 10
(5, 10)

julia> Flux.reset!(g);

julia> g(rand(Float32, 3))
5-element Vector{Float32}:
  0.05426188
 -0.111508384
  0.04700454
  0.06919164
  0.089212984

# A demonstration of not using `reset!` when the batch size changes.
julia> g = GRU(3, 5)
Recur(
  GRUCell(3, 5),                        # 140 parameters
)         # Total: 4 trainable arrays, 140 parameters,
          # plus 1 non-trainable, 5 parameters, summarysize 792 bytes.

julia> g.state |> size
(5, 1)

julia> g(rand(Float32, 3))
5-element Vector{Float32}:
 -0.11918676
 -0.089210495
  0.027523153
  0.017113047
  0.061968707

julia> g(rand(Float32, 3, 10)) # batch size of 10
5×10 Matrix{Float32}:
 -0.198102   -0.187499   -0.265959   -0.21598    -0.210867    -0.379202   -0.262658  -0.213773   -0.236976   -0.266929
 -0.138773   -0.137587   -0.208564   -0.155394   -0.142374    -0.289558   -0.200516  -0.154471   -0.165038   -0.198165
  0.040142    0.0716526   0.122938    0.0606727   0.00901341   0.0754129   0.107307   0.0551935   0.0366661   0.0648411
  0.0655876   0.0512702  -0.0813906   0.120083    0.0521291    0.175624    0.110025   0.0345626   0.189902   -0.00220774
  0.0756504   0.0913944   0.0982122   0.122272    0.0471702    0.228589    0.168877   0.0778906   0.145469    0.0832033

julia> g.state |> size # state shape has changed
(5, 10)

julia> g(rand(Float32, 3)) # outputs a length 5*10 = 50 vector.
50-element Vector{Float32}:
 -0.2639928
 -0.18772684
  ⋮
 -0.022745812
  0.040191136
```
"""
GRU(a...; ka...) = Recur(GRUCell(a...; ka...))
Recur(m::GRUCell) = Recur(m, m.state0)

# TODO remove in v0.13
function Base.getproperty(m::GRUCell, sym::Symbol)
  if sym === :h
    Zygote.ignore() do
      @warn "GRUCell field :h has been deprecated. Use m::GRUCell.state0 instead."
    end
    return getfield(m, :state0)
  else
    return getfield(m, sym)
  end
end


# GRU v3

struct GRUv3Cell{A,V,S}
  Wi::A
  Wh::A
  b::V
  Wh_h̃::A
  state0::S
end

GRUv3Cell(in, out; init = glorot_uniform, initb = zeros32, init_state = zeros32) =
  GRUv3Cell(init(out * 3, in), init(out * 2, out), initb(out * 3), 
            init(out, out), init_state(out,1))

function (m::GRUv3Cell{A,V,<:AbstractMatrix{T}})(h, x::Union{AbstractVecOrMat{T},OneHotArray}) where {A,V,T}
  b, o = m.b, size(h, 1)
  gx, gh, r, z = _gru_output(m.Wi, m.Wh, b, x, h)
  h̃ = tanh.(gate(gx, o, 3) .+ (m.Wh_h̃ * (r .* h)) .+ gate(b, o, 3))
  h′ = (1 .- z) .* h̃ .+ z .* h
  sz = size(x)
  return h′, reshape(h′, :, sz[2:end]...)
end

@functor GRUv3Cell

Base.show(io::IO, l::GRUv3Cell) =
  print(io, "GRUv3Cell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ")")

"""
    GRUv3(in::Integer, out::Integer)

[Gated Recurrent Unit](https://arxiv.org/abs/1406.1078v3) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences. This implements
the variant proposed in v3 of the referenced paper.

The parameters `in` and `out` describe the size of the feature vectors passed as input and as output. That is, it accepts a vector of length `in` or a batch of vectors represented as a `in x B` matrix and outputs a vector of length `out` or a batch of vectors of size `out x B`.

This constructor is syntactic sugar for `Recur(GRUv3Cell(a...))`, and so GRUv3s are stateful. Note that the state shape can change depending on the inputs, and so it is good to `reset!` the model between inference calls if the batch size changes. See the examples below.

See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.

# Examples
```jldoctest
julia> g = GRUv3(3, 5)
Recur(
  GRUv3Cell(3, 5),                      # 140 parameters
)         # Total: 5 trainable arrays, 140 parameters,
          # plus 1 non-trainable, 5 parameters, summarysize 848 bytes.

julia> g(rand(Float32, 3)) |> size
(5,)

julia> Flux.reset!(g);

julia> g(rand(Float32, 3, 10)) |> size # batch size of 10
(5, 10)

julia> Flux.reset!(g);

julia> g(rand(Float32, 3))
5-element Vector{Float32}:
  0.05637428
  0.0084088165
  -0.036565308
  0.013599886
  -0.0168455

# A demonstration of not using `reset!` when the batch size changes.
julia> g = GRUv3(3, 5)
Recur(
  GRUv3Cell(3, 5),                      # 140 parameters
)         # Total: 5 trainable arrays, 140 parameters,
          # plus 1 non-trainable, 5 parameters, summarysize 848 bytes.

julia> g.state |> size
(5, 1)

julia> g(rand(Float32, 3))
5-element Vector{Float32}:
  0.07569726
  0.23686615
  -0.01647649
  0.100590095
  0.06330994

julia> g(rand(Float32, 3, 10)) # batch size of 10
5×10 Matrix{Float32}:
  0.0187245   0.135969     0.0808607  0.138937    0.0153128   0.0386136  0.0498803  -0.0273552   0.116714    0.0584934
  0.207631    0.146397     0.226232   0.297546    0.28957     0.199815   0.239891    0.27778     0.132326    0.0325415
  0.083468   -0.00669185  -0.0562241  0.00725718  0.0319667  -0.021063   0.0682753   0.0109112   0.0188356   0.0826402
  0.0700071   0.120734     0.108757   0.14339     0.0850359   0.0706199  0.0915005   0.05131     0.105372    0.0507574
  0.0505043  -0.0408188    0.0170817  0.0190653   0.0936475   0.0406348  0.044181    0.139226   -0.0355197  -0.0434937

julia> g.state |> size # state shape has changed
(5, 10)

julia> g(rand(Float32, 3)) # outputs a length 5*10 = 50 vector.
50-element Vector{Float32}:
  0.08773954
  0.34562656
  ⋮
  0.13768406
  -0.015648054
```
"""
GRUv3(a...; ka...) = Recur(GRUv3Cell(a...; ka...))
Recur(m::GRUv3Cell) = Recur(m, m.state0)


@adjoint function Broadcast.broadcasted(f::Recur, args...)
  Zygote.∇map(__context__, f, args...)
end
