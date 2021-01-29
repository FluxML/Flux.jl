using Zygote: Params

Optimisers.init(o, xs::Params) = [init(o, x) for x in xs]

function Optimisers.update!(o, xs::Params, gs, states)
  for (i, (x, s)) in enumerate(zip(xs, states))
    _, states[i] = update!(o, x, gs[x], s)
  end

  return xs, states
end

# """
#     ClipValue(thresh)

# Clip gradients when their absolute value exceeds `thresh`.
# """
# mutable struct ClipValue{T}
#     thresh::T
# end

# apply!(o::ClipValue, x, Δ) = clamp!(Δ, -o.thresh, o.thresh)

# """
#     ClipNorm(thresh)

# Clip gradients when their L2 norm exceeds `thresh`.
# """
# mutable struct ClipNorm{T}
#     thresh::T
# end

# function apply!(o::ClipNorm, x, Δ)
#     Δnrm = norm(Δ)
#     if Δnrm > o.thresh
#         rmul!(Δ, o.thresh / Δnrm)
#     end
#     return Δ
# end
