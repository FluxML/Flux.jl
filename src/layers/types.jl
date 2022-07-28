import Adapt, Functors, Optimisers

"""
    Flux.AbstractLayer
    
Supertype for all of Flux's built-in layers.

Layer types are not essential to use your own `struct` with Flux.
But they do simplify some common interactions:

* Any `l::AbstractLayer` has a method for `Functors.functor`, 
  thus you need not invoke `@functor`. Note that your `struct` should
  have the default constructor. (Or something which similarly accepts all of
  its fields as arguments. It is simplest not to write an inner constructor.)

* Calling `Adapt.adapt` on any `l::AbstractLayer` will recurse using `Functors.fmap`,
  ensuring that the identification between tied weights is perserved.

* Subtypeing `PartialTrainLayer` marks only some fields as trainable,
  by overloading `Optimisers.trainable`.

* Some subtypes tell fancy `show` whether to unfold their contents: 
  `l::ContainerLayer` behaves like `Chain`, while `l::SimpleLayer` behaves like `Dense`.
"""
abstract type AbstractLayer end

function Functors.functor(::Type{T}, x) where {T<:AbstractLayer}
  if @generated
    F = fieldnames(T)
    args = map(sy -> :(getfield(x, $(QuoteNode(sy)))), F)
    C = Base.typename(T).name  # constructor
    recon = VERSION > v"1.9-" ? :(Splat($C)) : :(Base.splat($C))
    :((NamedTuple{$F}(($(args...),)), $recon))
  else
    # Getting this parameterless type takes about 2μs, every time:
    namedtuple(x), Base.splat(Base.typename(T).wrapper)
  end
end

function namedtuple(x::T) where T
  F = fieldnames(T)
  NamedTuple{F}(map(sy -> getfield(x, sy), F))
end

Adapt.adapt_structure(to, layer::AbstractLayer) = fmap(x -> adapt(to, x), layer)

"""
    Flux.ContainerLayer <: AbstractLayer
    
Supertype for layers such as `Chain` & `Parallel`. Not essential to Flux's functioning, 
but tells `show` to unfold the contents when this is the outermost struct.
And (like any `AbstractLayer`) removes the need for `@functor`.
"""
abstract type ContainerLayer <: AbstractLayer end
"""
    Flux.SimpleLayer <: AbstractLayer
    
Supertype for layers such as `Dense` & `Conv`. Not essential to Flux's functioning, 
but tells `show` how to behave. And (like any `AbstractLayer`) removes the need for `@functor`.
"""
abstract type SimpleLayer <: AbstractLayer end

"""
    Flux.PartialTrainLayer{which} <: SimpleLayer <: AbstractLayer

Supertype for layers such as `BatchNorm` which contain arrays of numbers
which are not to be optimised during training.

`which` is a tuple of `Symbol`s, indicating the fields of the struct
that that *do* contain trainable parameters. This is used by a method of
`Optimisers.trainable`, instead of writing that yourself.

Note that some fields (such as functions, integers, `nothing`) are never
trainable, and do not need special attention. `Optimisers.trainable` is needed
only to shield types which would otherwise be trainable, such as arrays of floats.

Also (like any `AbstractLayer`) removes the need for `@functor`,
and (like `SimpleLayer`) tells `show` not to unfold further.
"""
abstract type PartialTrainLayer{which} <: SimpleLayer end

function Optimisers.trainable(layer::PartialTrainLayer{which}) where {which}
  NamedTuple{which}(map(sy -> getfield(layer, sy), which))
end

"""
    Flux.NoTrainLayer <: SimpleLayer <: AbstractLayer

Supertype for layers which contain no trainable parameters.
"""
const NoTrainLayer = PartialTrainLayer{(;)}
