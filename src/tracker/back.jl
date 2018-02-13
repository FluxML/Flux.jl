init_grad(x) = zero(x)
zero_grad!(x) = zero(x)
zero_grad!(x::AbstractArray) = (x .= 0)

scan(c::Call) = foreach(scan, c.args)

function scan(x::Tracked)
  x.isleaf && return
  ref = x.ref += 1
  if ref == 1
    scan(x.f)
    isdefined(x, :grad) && (x.grad = zero_grad!(x.grad))
  else
    isdefined(x, :grad) || (x.grad = init_grad(x.data))
  end
  return
end

function scan(x)
  istracked(x) && scan(tracker(x))
  return
end

back_(f, y, args...) = back(f, args...)
back_(c::Call, y, Δ) = back_(c.func, y, Δ, c.args...)
back_(::Call{Void}, y, Δ) = nothing

accum!(x, Δ) = x .+ Δ
accum!(x::AbstractArray, Δ) = (x .+= Δ)

function back(x::Tracked, Δ)
  x.isleaf && (accum!(x.grad, Δ); return)
  ref = x.ref -= 1
  if isdefined(x, :grad)
    x.grad = accum!(x.grad, Δ)
    ref == 0 && back_(x.f, x.data, x.grad)
  else
    ref == 0 && back_(x.f, x.data, Δ)
  end
  return
end

back(x, Δ) = back(tracker(x), Δ)
back(x::Void, Δ) = error("Can't backpropagate through `nothing`")

macro back(x, Δ)
  quote
    x = $(esc(x))
    istracked(x) && back(x, $(esc(Δ)))
  end
end

# Interface methods

# TODO: if an error occurs in `back` the refcounts will be broken
# and `back` will silently fail to update.

function back!(x::Tracked, Δ)
  scan(x)
  back(x, Δ)
end

back!(x, Δ) = back!(tracker(x), Δ)
