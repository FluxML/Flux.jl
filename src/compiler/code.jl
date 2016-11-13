# TODO: proper escaping

import DataFlow: mapconst, cse

export @net

function process_func(ex, params = [])
  @capture(shortdef(ex), (args__,) -> body_)
  body = @> body MacroTools.flatten block liftloops(params) graphm DataFlow.il
  body = mapconst(x -> x in params ? :(self.$x) : x, body)
  return args, body
end

function makegraph(graph, args)
  @assert length(args) == 1
  mapconst(graph) do x
    x == args[1] ? inputnode(1) :
    isa(x, Offset) ? :(Offset($(Expr(:quote, x.name)), $(x.n), self.$(x.name))) :
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

import Lazy: groupby

reifyparams(v::IVertex) = mapconst(x -> isa(x, Param) ? x.x : x, v)

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
    (self::$T)($(args...),) = interpret(reifyparams(graph(self)), $(args...))
    update!(self::$T, η) = $(map(p -> :(update!(self.$p, η)), pnames)...)
    graph(self::$T) = $(DataFlow.constructor(makegraph(body, args)))
    nothing
  end |> esc
end

function process_anon(ex)
  args, body = process_func(ex)
  @assert length(args) == 1
  :(Flux.Capacitor($(DataFlow.constructor(makegraph(body, args))))) |> esc
end

macro net(ex)
  isexpr(ex, :type) ? process_type(ex) :
  isexpr(ex, :->, :function) ? process_anon(ex) :
  error("Unsupported model expression $ex")
end
