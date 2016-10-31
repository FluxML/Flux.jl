# TODO: most (all?) of this could be in DataFlow

immutable ModelInput end

splitnode(v, n) = vertex(Split(n), v)

inputnode(n) = splitnode(constant(ModelInput()), n)

isinput(v::IVertex) = isa(value(v), Split) && value(v[1]) == Constant(ModelInput())

function bumpinputs(v::IVertex)
  prewalk(v) do v
    isinput(v) ?
      inputnode(value(v).n + 1) :
      v
  end
end

function spliceinput(v::IVertex, input::IVertex)
  postwalk(v) do v
    value(v) == Constant(ModelInput()) ? input : v
  end
end

spliceinputs(v::IVertex, inputs::Vertex...) =
  spliceinput(v, vertex(Group(), inputs...))

function ninputs(v::IVertex)
  n = 0
  prewalk(v) do v
    isinput(v) && (n = max(n, value(v).n))
    v
  end
  return n
end

function detuple(v::IVertex)
  postwalk(v) do v
    if isa(value(v), Split) && isa(value(v[1]), Group)
      v[1][value(v).n]
    else
      v
    end
  end
end
