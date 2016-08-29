# TODO: proper escaping

import Flow: mapconst, cse

export @model

function process_func(ex, params = [])
  @capture(shortdef(ex), (args__,) -> body_)
  body = @> body MacroTools.flatten liftloops!(params) graphm Flow.il
  body = mapconst(x -> x in params ? :(self.$x) : x, body)
  return args, body
end

immutable ModelInput
  name
end

function makegraph(graph, args)
  @assert length(args) == 1
  mapconst(graph) do x
    x == args[1] ? ModelInput(1) :
    @capture(x, self.p_) ? ModelInput(p) :
      x
  end
end

function build_type(T, params)
  ex = quote
    type $T <: Model
      $(params...)
    end
  end
  if any(x->isexpr(x, Symbol), params)
    push!(ex.args,
      :($T($(map(x->isexpr(x, Symbol) ? :($x::AArray) : x, params)...)) =
          $T($(map(x->isexpr(x, Symbol) ? :(param($x)) : namify(x), params)...))))
  end
  ex
end

function deref_params(v)
  map(v) do x
    isa(x, Constant) && @capture(x.value, self.p_) ? Constant(:(state(self.$p))) : x
  end
end

function build_forward(body, args)
  iscyclic(body) && return :(error("Can't run forward pass on a cyclic graph"))
  syntax(cse(deref_params(body)))
end

function build_backward(body, x, params = [])
  iscyclic(body) && return :(error("Can't run backward pass on a cyclic graph"))
  Δs = invert(body)
  back = IVertex{Any}(Flow.Do())
  for param in params
    haskey(Δs, :(self.$param)) || continue
    ex = Δs[:(self.$param)]
    ex = deref_params(ex)
    thread!(back, @dvertex(accumulate!(:(self.$param), ex)))
  end
  ex = Δs[x]
  ex = deref_params(ex)
  thread!(back, @flow(tuple($ex)))
  syntax(cse(back))
end

function process_type(ex)
  @capture(ex, type T_ fs__ end)
  @destruct [params = false || [],
             funcs  = true || []] = groupby(x->isexpr(x, :->, :function), fs)
  @assert length(funcs) == 1
  pnames = namify.(params)
  args, body = process_func(funcs[1], pnames)
  @assert length(args) == 1
  quote
    $(build_type(T, params))
    (self::$T)($(args...),) = $(build_forward(body, args))
    back!(self::$T, Δ, $(args...)) = $(build_backward(body, args[1], pnames))
    update!(self::$T, η) = $(map(p -> :(update!(self.$p, η)), pnames)...)
    graph(::$T) = $(Flow.constructor(makegraph(body, args)))
    nothing
  end |> esc
end

function process_anon(ex)
  args, body = process_func(ex)
  layers = Set{Symbol}()
  Flow.prefor(body) do v
    isexpr(value(v), Symbol) && push!(layers, value(v))
  end
  @assert length(args) == 1
  :(Flux.Capacitor(
      ($(args...)) -> $(build_forward(body, args)),
      (Δ, $(args...)) -> $(build_backward(body, args[1])),
      η -> $(map(p -> :(update!($p, η)), layers)...),
      $(Flow.constructor(makegraph(body, args))))) |> esc
end

macro model(ex)
  isexpr(ex, :type) ? process_type(ex) :
  isexpr(ex, :->, :function) ? process_anon(ex) :
  error("Unsupported model expression $ex")
end
