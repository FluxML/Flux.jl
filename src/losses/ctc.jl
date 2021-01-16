using Flux
using Zygote: @adjoint
using Statistics
using NNlib

# CPU implementation
"""
  logaddexp(a, b)
Adds log-space `a` and `b` such that the result equals `log(exp(a)+exp(b))`
"""
function logaddexp(a, b)
  isinf(a) && return b
  isinf(b) && return a

  # always want the greater number on the left in the exponentiation;
  # the magnitude difference may end up making the number very positive
  # which will cause exp() to return Inf
  # E.g., a = -900, b = -800, will give exp(-800 - -900), which will be
  # Inf for Float32 values
  if a < b
    a, b = b, a
  end
  return a + log(1+exp(b-a))
end

"""
  add_blanks(z)

Adds blanks to the start and end of `z`, and between items in `z`
"""
function add_blanks(z, blank)
  z′ = fill(blank, 2*length(z) + 1)
  z′[2 .* eachindex(z)] = z
  return z′
end

function ctc_alpha(ŷ::AbstractArray, y)
  typed_zero = zero(ŷ[1])
  ŷ = logsoftmax(ŷ)
  blank = size(ŷ, 1)
  z′ = add_blanks(y, blank)
  T = size(ŷ, 2)
  U′ = length(z′)

  α = fill(log(typed_zero), U′, T)
  α[1,1] = ŷ[blank, 1]
  α[2,1] = ŷ[z′[2], 1]
  for t=2:T
	bound = max(1, U′ - 2(T - t) - 1)
    for u=bound:U′
	  if u == 1
		α[u,t] = α[u, t-1]
	  else
		α[u,t] = logaddexp(α[u, t-1], α[u-1, t-1])
		
		# array bounds check and f(u) function from Eq. 7.9
		if u > 2 && !(z′[u] == blank || z′[u-2] == z′[u])
		  α[u,t] = logaddexp(α[u,t], α[u-2,t-1])
		end
	  end
	  α[u,t] += ŷ[z′[u], t]
    end
  end
  return (loss=-1 * logaddexp(α[end,T], α[end-1, T]), alpha=α, zprime=z′, logsoftyhat=ŷ)
end
  
function ∇ctc_loss(ŷ::AbstractArray, y, out)
  loss, α, z′, ŷ = out
  U′, T = size(α)
  blank = size(ŷ, 1)
  typed_zero = zero(first(α))

  # Calculate beta coefficients, from the bottom-right, to the upper-left
  β = fill(log(typed_zero), U′, T)

  # Fill bottom-right corner so bounding errors can be avoided
  # by starting `u` at `U′-1`
  β[U′, T] = typed_zero
  β[U′-1, T] = typed_zero
  
  # start at T-1 so that β(T, u) = log(0) for all u < U′ - 1
  for t=(T-1):-1:1
	bound = min(U′, 2t)
    for u=bound:-1:1
	  if u == U′
		β[u,t] = ŷ[z′[u], t+1] + β[u, t+1]
	  else
		β[u,t] = logaddexp(ŷ[z′[u], t+1] + β[u, t+1], ŷ[z′[u+1], t+1] + β[u+1,t+1])

		# array bounds check and g(u) function from Eq. 7.16
		if u+2 <= U′ && z′[u] != blank && z′[u] != z′[u+2]
		  β[u,t] = logaddexp(β[u,t], ŷ[z′[u+2], t+1] + β[u+2, t+1])
	    end
	  end
    end
  end

  # Accumulate alpha-beta products for each category,
  # then calculate gradients
  accum = fill(log(typed_zero), size(ŷ))
  for t=1:T
    for u=1:U′
      accum[z′[u], t] = logaddexp(accum[z′[u], t], α[u,t] + β[u,t])
    end
  end
  grads = exp.(ŷ) .- exp.(accum .+ loss)
  return grads
end

"""
  ctc_loss(ŷ, y)

Computes the connectionist temporal classification loss between `ŷ`
and `y`.

`ŷ` must be a classes-by-time matrices, i.e., each row
represents a class and each column represents a time step.
Additionally, the `logsoftmax` function will be applied to `ŷ`, so
`ŷ` must be the raw activation values from the neural network and
not, for example, the activations after being passed through a
`softmax` activation function. `y` must be a 1D array of the labels
associated with `ŷ`. The blank label is assumed to be the last label
category in `ŷ`, so it is equivalent to `size(ŷ, 1)`.

Used for sequence-to-sequence classification problems such as
speech recognition and handwriting recognition where the exact
time-alignment of the output (e.g., letters) is not needed to
solve the problem. See [Graves et al. (2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
or [Graves (2012)](https://www.cs.toronto.edu/~graves/preprint.pdf#chapter.7)
for mathematical details.
"""
ctc_loss(ŷ::AbstractArray, y) = ctc_alpha(ŷ, y).loss

@adjoint function ctc_loss(ŷ, y)
	out = ctc_alpha(ŷ, y)
	ctc_loss_pullback(Δ) = (Δ .* ∇ctc_loss(ŷ, y, out), nothing)
	return out.loss, ctc_loss_pullback
end
