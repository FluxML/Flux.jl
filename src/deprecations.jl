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
