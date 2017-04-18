mutable struct Model <: Flux.Model
  graph
end

graphviz(m::Flux.Model) = Model(Flux.graph(m))
graphviz(m::Flux.Model, xs...; depth = 2) = graphviz(m)(xs...; depth=depth)

function (m::Model)(xs...; depth = 2)
  xs = map(rawbatch, xs)
  ctx = Context(mux(iline, ilambda, Flux.imap, iargs, ituple, interp),
                max_depth = depth, memory = 0, depth = 0, buffer = IOBuffer())
  interpret(ctx, m.graph, xs...)
  buffer = take!(ctx[:buffer]) |> String
  "digraph model {\n\n$buffer\n}", ctx[:memory]
end
