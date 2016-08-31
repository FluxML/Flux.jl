# TODO: change the input approach
immutable ModelInput
  name
end

isinput(x) = isa(x, Constant) && isa(x.value, ModelInput) && isa(x.value.name, Integer)

bumpinput(i::ModelInput) = isa(i.name, Integer) ? ModelInput(i.name + 1) : i
bumpinput(x) = x

bumpinputs(v::IVertex) = mapconst(bumpinput, v)

function spliceinputs(v::IVertex, inputs::IVertex...)
  postwalk(v) do v
    isinput(value(v)) ?
      inputs[value(v).value.name] :
      v
  end
end
