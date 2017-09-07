back!(c::Call, Δ) = back!(c.func, Δ, c.args...)
back!(::Call{Void}, Δ) = nothing

function back!(x::TrackedArray, Δ)
  isassigned(x.grad) && (x.grad[] .+= Δ)
  back!(x.f, Δ)
end

back!(x::TrackedScalar) = back!(x, 1)

macro back!(x, Δ)
  quote
    x = $(esc(x))
    istracked(x) && back!(x, $(esc(Δ)))
  end
end
