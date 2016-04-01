function delays(v::IVertex)
  ds = []
  Flow.prefor(v) do w
    value(w) == :Delay &&
      push!(ds, w)
  end
  return ds
end

function cut(v::IVertex, f = _ -> il(@flow(last(self.delay))))
  prewalk(v) do v
    value(v) == :Delay ? f(v) : v
  end
end

replaceall(d::Dict, args...) = Dict(k => replace(v, args...) for (k, v) in d)

# Create the forward function; a single delay node becomes an
# input and an output node.
function cut_forward(v::IVertex, params, ds = delays(v))
  pushes = map(x->vertex(:push!, vertex(:(self.delay)), x[1], map(vertex, params)...), ds)
  isempty(pushes) && return v
  @assert length(pushes) == 1
  v = vertex(Flow.Do(), pushes..., v)
  cut(v)
end

# Given a delay node, give the parameter gradients with respect to
# the node and a function which will propagate gradients around
# the loop.
function invertloop(v::IVertex, params)
  @gensym input
  v = cut(v[1], v -> vertex(input))
  Δs = invert(v, @flow(Δloop))
  Δs = replaceall(Δs, vertex(input), il(@flow(last(self.delay))))
  Δs, :((Δ, $input, $(params...)) -> $(syntax(cse(Δs[input]))))
end

# Returns:
#   Parameter gradients with respect to the function
#   Parameter gradients with respect to each delay node
function cut_backward(v::IVertex, params, ds = delays(v))
  isempty(ds) && return invert(v), []
  @assert length(ds) == 1
  @gensym input
  Δs = invert(cut(v, _ -> vertex(input)))
  Δs = replaceall(Δs, vertex(input), il(@flow(last(self.delay))))
  Δloop, ∇loop = invertloop(ds[1], params)
  Δh = vertex(:back!, vertex(:(self.delay)), Δs[input], vertex(∇loop))
  Δloop = replaceall(Δloop, vertex(:Δloop), Δh)
  Δs, [Δloop]
end

# g = il(@flow begin
#   hidden = σ( Wxh*x + Whh*Delay(hidden) + bh )
#   y = σ( Why*hidden + by )
# end)

# cut_backward(g, [:x])[1]
