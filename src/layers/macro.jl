
"""
    @layer [showtype] MyModel [trainable=(field1,...)]
 
This macro adds convenience functionality to a custom type to serve 
as a neural network layer, as a module, or as an entire model.

The optional keyword `trainable` allows you to specify which fields of your model can be trained, 
instead of assuming all `fieldnames(MyModel)` to trainable. 
Note that it is never necessary to tell Flux to ignore non-array objects such as functions or sizes.
This can be also be done by defining [`trainable(::MyModel)`](@ref Optimisers.trainable) for your type.

The macro also handles overloads of the 3-arg `show(::IO, ::MIME"text/plain", ::MyModel)` for pretty printing. 
The optional argument `showtype` can take any of the following values:

- `:expand` (default): This will expand the representation of container types like `Chain`, 
   while maintaining a compat representation of types like `Dense` containing only arrays.
- `:noexpand`: This is to be used in case your type contains other layers but you want to keep the representation simple.
- `:ignore`: To opt out of the pretty printing.

You probably still want to define 2-arg `show(::IO, ::MyModel)`, the macro does not touch this.

Note that re-running the macro with different options may not remove all methods, you will need to restart.

# Example
```jldoctest
julia> struct Trio; a; b; c end

julia> tri = Trio(Dense([1.1 2.2], [0.0], tanh), Dense(hcat(3.3), false), Dropout(0.4))
Trio(Dense(2 => 1, tanh), Dense(1 => 1; bias=false), Dropout(0.4))

julia> Flux.@layer Trio

julia> tri  # now the layer is printed like Chain
Trio(
  Dense(2 => 1, tanh),                  # 3 parameters
  Dense(1 => 1; bias=false),            # 1 parameters
  Dropout(0.4),
)                   # Total: 3 arrays, 4 parameters, 240 bytes.

julia> Flux.@layer :noexpand Trio trainable=(a,b)

julia> tri  # now the layer is printed compactly
Trio(Dense(2 => 1, tanh), Dense(1 => 1; bias=false), Dropout(0.4))  # 4 parameters

julia> opt_state = Flux.setup(Adam(), tri); # `c` is not in the optimizer state
```

The macro also adds methods to make using Flux with Enzyme easier.
* `Duplicated(m::Layer)` allocates a copy for the gradient (initially zero).
* This is made callable, `(m::Duplicated{<:Layer})(x...) = m.val(x...)`
* Pretty printing for `show(io, mime, ::Duplicated{<:Layer})`

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
  elseif exs[1] == QuoteNode(:noexpand)
    push!(out.args, _macro_layer_show(esc(exs[2])))
    exs[2:end]
  elseif exs[1] == QuoteNode(:ignore)
    exs[2:end]
  elseif exs[1] isa QuoteNode
    error("`@layer` accepts only the options `:ignore`, `:noexpand`, and `:expand` before the layer type (to control `show`).")
  else
    push!(out.args, _macro_big_show(esc(exs[1])))
    exs
  end
  
  push!(out.args, _macro_adapt(esc(type)))

  push!(out.args, _macro_enzyme(esc(type)))

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

function _macro_enzyme(type)
  out = quote
    # One-arg method Duplicated(m::Layer) which allocates & zeros the gradient:
    # Remove once https://github.com/EnzymeAD/Enzyme.jl/pull/2118 is merged
    $EnzymeCore.Duplicated(m::$type) = $EnzymeCore.Duplicated(m, $EnzymeCore.make_zero(m))

    # Make Duplicated{<:Layer} callable:
    function (m::$EnzymeCore.Duplicated{<:$type})(xs...)
        Zygote.isderiving() && error("""`Duplicated(flux_model)` is only for use with Enzyme.jl.
            `Flux.gradient` should detect this, but calling `Zygote.gradient` directly on
            such a wrapped model is not supported.""")
        m.val(xs...)
    end

    # Not sure but this does prevent printing of 2nd copy:
    $Optimisers.trainable(m::$EnzymeCore.Duplicated{<:$type}) = (; val = m.val)
  end
  # Add a show method for Duplicated{<:Layer}
  push!(out.args, _macro_big_show(:($EnzymeCore.Duplicated{<:$type})))
  out
end

function _show_pre_post(obj::EnzymeCore.Duplicated)
  nrm = norm(destructure(obj.dval)[1])
  str = repr(round(nrm; sigdigits=3))
  "Duplicated(", "  # norm(∇) ≈ $str\n) "
end
