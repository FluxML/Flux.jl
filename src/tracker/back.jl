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

back_(f, y, args...) = back(f, args...)
back_(c::Call, y, Δ) = back_(c.func, y, Δ, c.args...)
back_(::Call{Void}, y, Δ) = nothing

function back(x::TrackedArray, Δ)
  ref = x.ref -= 1
  if isdefined(x, :grad)
    x.grad .+= Δ
    ref == 0 && back_(x.f, x.data, x.grad)
  else
    ref == 0 && back_(x.f, x.data, Δ)
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

# TODO: if an error occurs in `back` the refcounts will be broken
# and `back` will silently fail to update.

function back!(x::TrackedArray, Δ)
  scan(x)
  back(x, Δ)
end

back!(x::TrackedScalar) = back!(x, 1)
