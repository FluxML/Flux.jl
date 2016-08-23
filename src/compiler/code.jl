import Flow: mapconst, cse

function process_func(ex, params)
  @capture(shortdef(ex), (args__,) -> body_)
  body = Flow.il(graphm(body))
  body = mapconst(x -> x in params ? :(self.$x) : x, body)
  return args, body
end

immutable Parameter
  name
end

function makegraph(graph, args)
  @assert length(args) == 1
  mapconst(graph) do x
    x == args[1] ? Parameter(1) :
    @capture(x, self.p_) ? Parameter(p) :
      x
  end
end

function build_type(T, params)
  quote
    type $T <: Model
      $(params...)
    end
    $T($(map(x->:($x::AArray), params)...)) = $T($(map(x->:(param($x)), params)...))
  end
end

function deref_params(v)
  mapconst(v) do x
    @capture(x, self.p_) ? :(state(self.$p)) : x
  end
end

function build_forward(body, args)
  cse(deref_params(body))
end

function build_backward(body, x, params)
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
  cse(back)
end

function process_type(ex)
  @capture(ex, type T_ fs__ end)
  @destruct [params = false || [],
             funcs  = true || []] = groupby(x->isexpr(x, :->, :function), fs)
  @assert length(funcs) == 1
  args, body = process_func(funcs[1], params)
  @assert length(args) == 1
  quote
    $(build_type(T, params))
    (self::$T)($(args...),) = $(syntax(build_forward(body, args)))
    back!(self::$T, Δ, $(args...)) = $(syntax(build_backward(body, args[1], params)))
    update!(self::$T, η) = $(map(p -> :(update!(self.$p, η)), params)...)
    graph(::$T) = $(Flow.constructor(makegraph(body, args)))
    nothing
  end |> esc
end

macro model(ex)
  isexpr(ex, :type) ? process_type(ex) :
  error("Unsupported model expression $ex")
end
