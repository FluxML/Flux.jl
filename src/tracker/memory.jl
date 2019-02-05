function mem(x::Array, seen)
  x in seen && return 0
  push!(seen, x)
  return sizeof(x)
end

fields(x) = map(f -> isdefined(x, f) ? getfield(x, f) : nothing, fieldnames(typeof(x)))

function mem(x, seen)
  isbits(x) && return sizeof(x)
  x in seen && return 0
  push!(seen, x)
  sum(x -> mem(x, seen), fields(x))
end

mem(x) = mem(x, IdSet())

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  function mem(x::CuArray, seen)
    x in seen && return 0
    push!(seen, x)
    return sizeof(x)
  end
end
