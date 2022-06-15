# v0.12 deprecations

function ones(dims...)
  Base.depwarn("Flux.ones(size...) is deprecated, please use Flux.ones32(size...) or Base.ones(Float32, size...)", :ones, force=true)
  Base.ones(Float32, dims...)
end
ones(T::Type, dims...) = Base.ones(T, dims...)

function zeros(dims...)
  Base.depwarn("Flux.zeros(size...) is deprecated, please use Flux.zeros32(size...) or Base.zeros(Float32, size...)", :zeros, force=true)
  Base.zeros(Float32, dims...)
end
zeros(T::Type, dims...) = Base.zeros(T, dims...)

ones32(::Type, dims...) = throw(ArgumentError("Flux.ones32 is always Float32, use Base.ones to specify the element type"))
zeros32(::Type, dims...) = throw(ArgumentError("Flux.zeros32 is always Float32, use Base.zeros to specify the element type"))

# v0.13 deprecations

function Broadcast.broadcasted(f::Recur, args...)
  # This had an explicit @adjoint rule, calling Zygote.∇map(__context__, f, args...), until v0.12
  Base.depwarn("""Broadcasting is not safe to use with RNNs, as it does not guarantee an iteration order.
    Re-writing this as a comprehension would be better.""", :broadcasted)
  map(f, args...)  # map isn't really safe either, but 
end

@deprecate frequencies(xs) group_counts(xs)

struct Zeros
  function Zeros()
    Base.depwarn("Flux.Zeros is no more, has ceased to be, is bereft of life, is an ex-boondoggle... please use bias=false instead", :Zeros)
    false
  end
end
Zeros(args...) = Zeros()  # was used both Dense(10, 2, initb = Zeros) and Dense(rand(2,10), Zeros())

function Optimise.update!(x::AbstractArray, x̄)
  Base.depwarn("`Flux.Optimise.update!(x, x̄)` was not used internally and has been removed. Please write `x .-= x̄` instead.", :update!)
  x .-= x̄
end

function Diagonal(size::Integer...; kw...)
  Base.depwarn("Flux.Diagonal is now Flux.Scale, and also allows an activation function.", :Diagonal)
  Scale(size...; kw...)
end
function Diagonal(size::Tuple; kw...)
  Base.depwarn("Flux.Diagonal is now Flux.Scale, and also allows an activation function.", :Diagonal)
  Scale(size...; kw...)
end

# Deprecate this eventually once saving models w/o structure is no more
function loadparams!(m, xs)
  Base.depwarn("loadparams! will be deprecated eventually. Use loadmodel! instead.", :loadparams!)
  for (p, x) in zip(params(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(p, x)
  end
end

# Channel notation: Changed to match Conv, but very softly deprecated!
# Perhaps change to @deprecate for v0.14, but there is no plan to remove these.
Dense(in::Integer, out::Integer, σ = identity; kw...) =
  Dense(in => out, σ; kw...)
Bilinear(in1::Integer, in2::Integer, out::Integer, σ = identity; kw...) =
  Bilinear((in1, in2) => out, σ; kw...)
Embedding(in::Integer, out::Integer; kw...) = Embedding(in => out; kw...)

RNNCell(in::Integer, out::Integer, σ = tanh; kw...) = RNNCell(in => out, σ; kw...)
LSTMCell(in::Integer, out::Integer; kw...) = LSTMCell(in => out; kw...)

GRUCell(in::Integer, out::Integer; kw...) = GRUCell(in => out; kw...)
GRUv3Cell(in::Integer, out::Integer; kw...) = GRUv3Cell(in => out; kw...)

# Optimisers with old naming convention
@deprecate ADAM Adam
@deprecate NADAM NAdam
@deprecate ADAMW AdamW
@deprecate RADAM RAdam
@deprecate OADAM OAdam
@deprecate ADAGrad AdaGrad
@deprecate ADADelta AdaDelta
