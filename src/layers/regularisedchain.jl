"""
    RegularisedChain(layers,regularizations)
    RegularisedChain(regularization,layers...)

Chain multiple layers / functions together, so that they are called in sequence
on a given input.

If the chain is called with additional scalar parameter, appropriate regulariser is 
called on output of each layer. Regularised chain then returns Tuple, where the first 
item is the output of the Chain and the second is the value of regularization.

An example where this construct might be useful is 
Generalization in Deep Learning, Kenji Kawaguchi, Leslie Pack Kaelbling, Yoshua Bengio, 2017
https://arxiv.org/abs/1710.05468

```julia
import Flux: RegularisedChain, nullregularizer, l1
m = RegularisedChain([Dense(10, 5), Dense(5, 2)],[nullregularizer,l1])
m = Flux.RegularisedChain(Flux.l1,Dense(10, 5), Dense(5, 2))
x = rand(10)
m(x) == m[2](m[1](x)) # invocation equals that of the normal chain
m(x,0)								# invocation with regularization, returns tuple where first item is equal to m(x)
m(x,0)[1] == m(x)
```

`RegularisedChain` also supports indexing and slicing, e.g. `m[2]` or `m[1:end-1]`.
`m[1:3](x)` will calculate the output of the first three layers.
"""
type RegularisedChain
  layers::Vector{Any}
  regularisers::Vector{Any}
  RegularisedChain(layers,regularisers)  = begin 
  	assert(length(layers) == length(regularisers))
  	new(layers,regularisers)
  end
end

RegularisedChain(regulariser,layers...) = RegularisedChain([layers...],fill(regulariser,length(layers)))

@forward RegularisedChain.layers Base.getindex, Base.first, Base.last, Base.endof, Base.push!
@forward RegularisedChain.layers Base.start, Base.next, Base.done

children(c::RegularisedChain) = vcat(c.layers,c.regularisers)
mapchildren(f, c::RegularisedChain) = RegularisedChain(f.(c.layers)...,f.(c.regularisers))
adapt(T, c::RegularisedChain) = RegularisedChain(map(x -> adapt(T, x), c.layers),map(x -> adapt(T, x), c.regularisers))

function applyregulariser(x,m)
	o = m[1](x[1])
	o, x[2] + m[2](o) 
end
(c::RegularisedChain)(x) = foldl((x, m) -> m(x), x, c.layers)
(c::RegularisedChain)(x::Tuple) = foldl((x, m) -> applyregulariser(x,m), x, zip(c.layers,c.regularisers))
(c::RegularisedChain)(x,l) = c((x,l))

Base.getindex(c::RegularisedChain, i::AbstractArray) = RegularisedChain(c.layers[i],c.regularisers[i])

function Base.show(io::IO, c::RegularisedChain)
  print(io, "RegularisedChain(")
  join(io, zip(c.layers,c.regularisers), ", ")
  print(io, ")")
end


nullregularizer(x) = 0
l2(x) = mean(x.^2)
l1(x) = mean(abs.(x))
function regcov(x)
  xx = x .- mean(x,2)
  mean(xx*transpose(xx))
end
function logcov(x)
  o = x*transpose(x)
  mean(o) - mean(log.(diag(o) .+ 1f-4 ))
end
#Generalization in Deep Learning, Kenji Kawaguchi, Leslie Pack Kaelbling, Yoshua Bengio
function darc1(x)
  l = sum(abs.(x),2)/size(x,2)
  l[indmax(Flux.Tracker.data(l))]
end