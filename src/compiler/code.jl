import DataFlow: cse
using MacroTools: @q

export @net

function graphdef(ex, params = [])
  @capture(shortdef(ex), (args__,) -> body_)
  body = @> body MacroTools.flatten liftloops graphm DataFlow.il
  body = map(x -> x in params ? :(self.$x) : x, body)
  return args, body
end

function makegraph(graph, args, params = [])
  graph = prewalk(graph) do v
    isconstant(v) && (i = findfirst(args, value(v[1]))) ≠ 0 ?
      inputnode(i) :
      v
  end
  graph = map(graph) do x
    x isa Offset ?
      :(Flux.Offset($(Expr(:quote, x.name)), $(x.n),
                    $(x.name in params ? :(self.$(x.name)) : x.name))) :
      x
  end
  vertex(:(Flux.Frame(self)), graph)
end

function build_type(T, params)
  @esc T
  ex = quote
    type $T
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
    @capture(x, self.p_) ? :(Flux.state(self.$p)) : x
  end
end

function build_forward(body, args)
  iscyclic(body) && return :(error("Can't run forward pass on a cyclic graph"))
  applylines(syntax(cse(deref_params(body))))
end

import Lazy: groupby

reifyparams(v::IVertex) = map(x -> x isa Param ? x.x : x, v)

# TODO: type hints for parameters

function process_type(ex)
  @capture(ex, type T_ fs__ end)
  @destruct [params = false || [],
             funcs  = true || []] = groupby(x->isexpr(x, :->, :function), fs)
  @assert length(funcs) == 1
  pnames = namify.(params)
  args, body = graphdef(funcs[1], pnames)
  self = esc(:self)
  quote
    $(build_type(T, params))
    $(esc(:((self::$T)($(args...)) = $(build_forward(body, args)))))
    $(esc(:(Flux.update!(self::$T, η)))) = ($(map(p -> :(update!($self.$p, η)), pnames)...);)
    $(esc(:(Flux.graph(self::$T)))) = $(DataFlow.constructor(map(esc, makegraph(body, args, params))))
    nothing
  end
end

function process_anon(ex)
  args, body = graphdef(ex)
  :(Capacitor($(DataFlow.constructor(map(esc, makegraph(body, args)[1])))))
end

function process_def(ex)
  # TODO: make a singleton net type
  @capture(ex, f_(xs__) = body_)
  :($(esc(f)) = @net $(esc(:(($(xs...),) -> $body))); nothing)
end

macro net(ex)
  ex = shortdef(ex)
  isexpr(ex, :type) ? process_type(ex) :
  @capture(ex, (__,) -> _) ? process_anon(ex) :
  @capture(ex, _(__) = _) ? process_def(ex) :
  error("Unsupported model expression $ex")
end
