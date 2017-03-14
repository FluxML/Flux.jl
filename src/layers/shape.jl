export Input

Dims{N} = NTuple{N,Int}

dims(d::Dims) = d

dims(i...) = (i...,)

single(i) = i
single(i::Dims) = length(i) == 1 ? first(i) : i

# Shim for kicking off shape inference

type ShapeError <: Exception
  layer
  shape
end

type Input{N} <: Model
  dims::Dims{N}
end

Input(i...) = Input(dims(i...))

(::Input)(x) = x
back!(::Input, Δ, x) = Δ

# Initialise placeholder

type Init{F}
  f::F
end

init(i::Init, input...) = i.f(input...)
init(m, input...) = m

# Shape inference API

shape(x, in) = in

shape(i::Input, _) = i.dims

# Implementation for bundled layers

shape(d::Affine, _) = length(state(d.b)) # TODO: could perhaps infer this

Affine(out::Integer) = Init(in::Integer -> Affine(in, out))
