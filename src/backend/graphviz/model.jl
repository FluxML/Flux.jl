mutable struct Model <: Flux.Model
  graph
end

graphviz(m::Flux.Model) = Model(Flux.graph(m))
graphviz(m::Flux.Model, xs...) = graphviz(m)(xs...)

function (m::Model)(xs...)
  xs = map(rawbatch, xs)
  ctx = Context(mux(iline, ilambda, Flux.imap, iargs, ituple, interp),
                memory = 0, depth = 0, buffer = IOBuffer())
  interpret(ctx, m.graph, xs...)
  buffer = take!(ctx[:buffer]) |> String
  "digraph model {\n\n$buffer\n}", ctx[:memory]
end
