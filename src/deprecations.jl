# v0.12 deprecations

function ones(dims...)
  Base.depwarn("Flux.ones(size...) is deprecated, please use Flux.ones32(size...) or Base.ones(Float32, size...)", :ones)
  Base.ones(Float32, dims...)
end
ones(T::Type, dims...) = Base.ones(T, dims...)

function zeros(dims...)
  Base.depwarn("Flux.zeros(size...) is deprecated, please use Flux.zeros32(size...) or Base.zeros(Float32, size...)", :zeros)
  Base.zeros(Float32, dims...)
end
zeros(T::Type, dims...) = Base.zeros(T, dims...)

ones32(::Type, dims...) = throw(ArgumentError("Flux.ones32 is always Float32, use Base.ones to specify the element type"))
zeros32(::Type, dims...) = throw(ArgumentError("Flux.zeros32 is always Float32, use Base.zeros to specify the element type"))

# v0.13 deprecations

@deprecate frequencies(xs) group_counts(xs)

# Channel notation: Changed to match Conv, but very softly deprecated!
# Perhaps change to @deprecate for v0.14, but there is no plan to remove these.
Dense(in::Integer, out::Integer, σ = identity; kw...) =
  Dense(in => out, σ; kw...)
Bilinear(in1::Integer, in2::Integer, out::Integer, σ = identity; kw...) =
  Bilinear((in1, in2) => out, σ; kw...)
Embedding(in::Integer, out::Integer; kw...) = Embedding(in => out; kw...)

