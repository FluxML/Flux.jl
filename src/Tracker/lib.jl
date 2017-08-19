import Base: *

a::TrackedMatrix * b::Union{TrackedMatrix,AbstractMatrix} = Var(Call(*, a, b))
a::Union{TrackedMatrix,AbstractMatrix} * b::TrackedMatrix = Var(Call(*, a, b))

function back!(::typeof(*), Δ, a::AbstractArray, b::AbstractArray)
  back!(a, A_mul_Bt(Δ, data(b)))
  back!(b, At_mul_B(data(a), Δ))
  return
end
