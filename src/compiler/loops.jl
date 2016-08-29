type Delay
  name::Symbol
end

function liftloops!(ex, params)
  e = Flow.normedges(ex)
  hidden = intersect((b.args[1] for b in ex.args), params)
  edges = Dict(h => gensym("edge") for h in hidden)
  for b in ex.args
    b.args[2] = MacroTools.postwalk(x -> get(edges, x, x), b.args[2])
  end
  for (h, e) in edges
    unshift!(ex.args, :($e = $(Delay(h))($h)))
  end
  return ex
end

bumpinput(i::ModelInput) = isa(i.name, Integer) ? ModelInput(i.name + 1) : i
bumpinput(x) = x

bumpinputs(v::IVertex) = mapconst(bumpinput, v)

function break!(model)
  iscyclic(graph(model)) || return model
  bumpinputs(graph(model))
end

# r = Recurrent(784, 10, 50)

# break!(r)

@model type Recurrent
  Wxh; Whh; Bh
  Wxy; Why; By
  hidden

  function (x)
    hidden = σ( Wxh*x + Whh*hidden + Bh )
    y = σ( Wxy*x + Why*hidden + By )
  end
end

Recurrent(in::Integer, out::Integer, hidden::Integer; init = initn) =
  Recurrent(init(hidden, in), init(hidden, hidden), init(hidden),
            init(out, in), init(out, hidden), init(hidden),
            zeros(hidden))

Base.show(io::IO, r::Recurrent) =
  print(io, "Flux.Recurrent(...)")
