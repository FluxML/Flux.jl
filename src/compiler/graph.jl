# TODO: change the input approach
immutable ModelInput
  n::Int
end

isinput(x) = isa(x, Constant) && isa(x.value, ModelInput)

bumpinput(i::ModelInput) = ModelInput(i.n + 1)
bumpinput(x) = x

bumpinputs(v::IVertex) = mapconst(bumpinput, v)

function spliceinputs(v::IVertex, inputs::IVertex...)
  postwalk(v) do v
    isinput(value(v)) ?
      inputs[value(v).value.n] :
      v
  end
end
