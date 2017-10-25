scan(x) = nothing

scan(c::Call) = foreach(scan, c.args)

function scan(x::TrackedArray)
  ref = x.ref += 1
  if ref == 1
    scan(x.f)
  else
    isdefined(x, :grad) || (x.grad = zeros(x.data))
  end
  return
end

back(c::Call, Δ) = back(c.func, Δ, c.args...)
back(::Call{Void}, Δ) = nothing

function back(x::TrackedArray, Δ)
  ref = x.ref -= 1
  if isdefined(x, :grad)
    x.grad .+= Δ
    ref == 0 && back(x.f, x.grad)
  else
    ref == 0 && back(x.f, Δ)
  end
  return
end

macro back(x, Δ)
  quote
    x = $(esc(x))
    istracked(x) && back(x, $(esc(Δ)))
  end
end

# Interface methods

function back!(x::TrackedArray, Δ)
  scan(x)
  back(x, Δ)
end

back!(x::TrackedScalar) = back!(x, 1)
