import DataFlow: mapconst, cse
using MacroTools: @q

export @net, @ml

function process_func(ex, params = [])
  @capture(shortdef(ex), (args__,) -> body_)
  body = @> body MacroTools.flatten liftloops graphm DataFlow.il
  body = mapconst(x -> x in params ? :(self.$x) : x, body)
  return args, body
end

function makegraph(graph, args)
  graph = prewalk(graph) do v
    value(v) isa Constant && (i = findfirst(x->x==value(v).value, args)) ≠ 0 ?
      inputnode(i) :
      v
  end
  graph = map(graph) do x
    x isa Offset ?
      :(Flux.Offset($(Expr(:quote, x.name)), $(x.n), self.$(x.name))) :
      x
  end
  vertex(:(Flux.Frame(self)), graph)
end

function build_type(T, params)
  @esc T
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

runmodel(f, xs...) = f(xs...)

function deref_params(v)
  v = map(v) do x
    x isa Constant && @capture(x.value, self.p_) ? Constant(:(Flux.state(self.$p))) : x
  end
  prewalk(v) do v
    @capture(value(v), self.p_) ? vertex(:(Flux.runmodel), constant(:(self.$p)), inputs(v)...) : v
  end
end

function build_forward(body, args)
  iscyclic(body) && return :(error("Can't run forward pass on a cyclic graph"))
  applylines(syntax(cse(deref_params(body))))
end

import Lazy: groupby

reifyparams(v::IVertex) = mapconst(x -> x isa Param ? x.x : x, v)

# TODO: type hints for parameters

function process_type(ex)
  @capture(ex, type T_ fs__ end)
  @destruct [params = false || [],
             funcs  = true || []] = groupby(x->isexpr(x, :->, :function), fs)
  @assert length(funcs) == 1
  pnames = namify.(params)
  args, body = process_func(funcs[1], pnames)
  self = esc(:self)
  quote
    $(build_type(T, params))
    $(@q $(esc(:(Flux.runmodel(self::$T, $(args...)) = $(build_forward(body, args))))))
    ($self::$(esc(T)))($(args...)) = runrawbatched((xs...) -> runmodel($self, xs...), $(args...))
    $(esc(:(Flux.update!(self::$T, η)))) = ($(map(p -> :(update!($self.$p, η)), pnames)...);)
    $(esc(:(Flux.graph(self::$T)))) = $(DataFlow.constructor(mapconst(esc, makegraph(body, args))))
    nothing
  end
end

function process_anon(ex)
  args, body = process_func(ex)
  @assert length(args) == 1
  :(Capacitor($(DataFlow.constructor(mapconst(esc, makegraph(body, args)[1])))))
end

macro net(ex)
  ex = shortdef(ex)
  isexpr(ex, :type) ? process_type(ex) :
  @capture(ex, (__,) -> _) ? process_anon(ex) :
  @capture(ex, _(__) = _) ? error("@net functions not implemented") :
  error("Unsupported model expression $ex")
end
