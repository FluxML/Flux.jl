
"""
    @layer Dense
    @layer :expand Chain
    @layer BatchNorm trainable=(β,γ)

This macro adds convenience functionality to a custom type to serve 
as a neural network layer, module, or entire model.

The keyword `trainable` allows you to limit this exploration, instead of visiting all `fieldnames(T)`.
Note that it is never necessary to tell Flux to ignore non-array objects such as functions or sizes.

The macro also handles overloads of `show` for pretty printing.
* By default, it adds methods to 3-arg `Base.show` to treat your layer much like `Dense` or `Conv`.
* If your layer is a container, more like `Chain` or `Parallel`, then `:expand` makes `show` unfold its contents.
* To disable all `show` overloads, there is an `:ignore` option too.

(You probably still want to define 2-arg `show(io::IO, x::Layer)`, the macro does not touch this.)

Note that re-running the macro with different options may not remove all methods, you will need to restart.

# Example
```jldoctest
julia> struct Trio; a; b; c end

julia> tri = Trio(Dense([1.1 2.2], [0.0], tanh), Dense(hcat(3.3), false), Dropout(0.4))
Trio(Dense(2 => 1, tanh), Dense(1 => 1; bias=false), Dropout(0.4))

julia> Flux.@layer :expand Trio

julia> tri  # now the layer is printed like Chain
Trio(
  Dense(2 => 1, tanh),                  # 3 parameters
  Dense(1 => 1; bias=false),            # 1 parameters
  Dropout(0.4),
)                   # Total: 3 arrays, 4 parameters, 240 bytes.
```

"""
macro layer(exs...)
  _layer_macro(exs...)
end

function _layer_macro(exs...)
  out = quote end

  # These functions are defined in show.jl, and each return an expression overloading Base.show
  type, rest... = if exs[1] == QuoteNode(:expand)
    push!(out.args, _macro_big_show(esc(exs[2])))  
    exs[2:end]
  elseif exs[1] == QuoteNode(:ignore)
    exs[2:end]
  elseif exs[1] isa QuoteNode
    error("`@layer` accepts only two options before the layer type, `:expand` and `:ignore` (to control `show`)")
  else
    push!(out.args, _macro_layer_show(esc(exs[1])))
    exs
  end
  
  push!(out.args, _macro_adapt(esc(type)))

  for j in 1:length(rest)
    ex = rest[j]
    Meta.isexpr(ex, :(=)) || error("The macro `@layer` expects here `keyword = (fields...,)`, got ", ex)

    name = if ex.args[1] == :trainable
      :(Optimisers.trainable)
    else
      error("`@layer` cannot define a method for `$(ex.args[1])` at the moment, sorry.")
      # @warn "Trying to define a method for `$(ex.args[1])` in your scope... this is experimental" maxlog=1
      # esc(ex.args[1])
    end
    push!(out.args, _macro_trainable(esc(type), name, ex.args[2]))
  end

  return out
end

# @layer's code for Adapt
function _macro_adapt(type)
  quote
    Adapt.adapt_structure(to, layer::$type) = $fmap($adapt(to), layer)
  end
end

# @layer's code for Optimisers.trainable, and perhaps anything else,
# with the pattern that keywords mean function names & what fields they pick.

function _macro_trainable(type, fun, fields)
  Meta.isexpr(fields, :tuple) || error("expected a tuple of field names")
  symbols = Tuple(map(_noquotenode, fields.args))
  quoted = map(QuoteNode, symbols)
  gets = [:(getfield(x, $f)) for f in quoted]
  quote
    $fun(x::$type) = NamedTuple{$symbols}(($(gets...),))
  end
end
_macro_trainable(type, fun, field::Union{Symbol,QuoteNode}) = _macro_trainable(type, fun, :(($field,)))  # lets you forget a comma

_noquotenode(s::Symbol) = s
_noquotenode(q::QuoteNode) = q.value  # lets you write trainable=(:x,:y) instead of (x,y)
_noquotenode(ex) = error("expected a symbol here, as a field name, but got ", ex)
