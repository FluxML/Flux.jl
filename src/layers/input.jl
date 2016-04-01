export Input

typealias Dims{N} NTuple{N,Int}

dims(d::Dims) = d

dims(i...) = (i...,)

type Input{N} <: Model
  dims::Dims{N}
end

Input(i) = Input(dims(i))

(::Input)(x) = x
back!(::Input, ∇) = ∇

shape(i::Input) = i.dims

# Initialise placeholder

type Init{F}
  f::F
end

(f::Init)(args...) = f.f(args...)
