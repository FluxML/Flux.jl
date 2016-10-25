immutable ModelInput end

inputnode(n) = vertex(Split(n), constant(ModelInput()))

function bumpinputs(v::IVertex)
  prewalk(v) do v
    isa(value(v), Split) && value(v[1]) == Constant(ModelInput()) ?
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

function detuple(v::IVertex)
  postwalk(v) do v
    if isa(value(v), Split) && isa(value(v[1]), Group)
      v[1][value(v).n]
    else
      v
    end
  end
end
