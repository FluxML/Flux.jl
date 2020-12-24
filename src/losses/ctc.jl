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
  F(A, blank)

Removes blanks and repetitions in the sequence `A`

This is the function `F` as defined in Graves (2012)
"""
function F(A, blank)
  prev = A[1]
  z = [prev]
  for curr in A[2:end]
    if curr != prev && curr != blank
      push!(z, curr)
    end
    prev = curr
  end
  return z
end

"""
  add_blanks(z)

Adds blanks to the start and end of `z`, and between item in `z`
"""
function add_blanks(z, blank)
  z′ = [blank]
  for label in z
    push!(z′, label)
    push!(z′, blank)
  end
  return z′
end

function ctc_(ŷ, y)
  typed_zero = zero(ŷ[1])
  ŷ = logsoftmax(ŷ)
  blank = size(ŷ, 1)
  z = F(Base.argmax.(eachcol(y)), blank)
  z′ = add_blanks(z, blank)
  T = size(ŷ, 2)
  U′ = length(z′)

  α = fill(log(typed_zero), U′, T)
  α[1,1] = ŷ[blank, 1]
  α[2,1] = ŷ[z′[2], 1]
  for t=2:T
	bound = U′ - 2(T - t) - 1
    for u=1:U′
	  u < bound && continue
	  if u == 1
		α[u,t] = α[u, t-1]
	  else
		α[u,t] = logaddexp(α[u, t-1], α[u-1, t-1])
		
		# array bounds check and f(u) function from Eq. 7.9
		if z′[u] != blank && u != 2 && z′[u] != z′[u-2]
		  α[u,t] = logaddexp(α[u,t], α[u-2,t-1])
		end
	  end
	  α[u,t] += ŷ[z′[u], t]
    end
  end

  # Calculate beta coefficients, from the bottom-right, to the upper-left
  β = fill(log(typed_zero), U′, T)

  # Fill bottom-right corner so bounding errors can be avoided
  # by starting `u` at `U′-1`
  β[U′, T] = typed_zero
  β[U′-1, T] = typed_zero
  
  # start at T-1 so that β(T, u) = log(0) for all u < U′ - 1
  for t=(T-1):-1:1
	bound = min(U′ + 1, 2t)
    for u=U′:-1:1
      u > bound && continue
	  if u == U′
		β[u,t] = ŷ[z′[u], t+1] + β[u, t+1]
	  else
		β[u,t] = logaddexp(ŷ[z′[u], t+1] + β[u, t+1], ŷ[z′[u+1], t+1] + β[u+1,t+1])
		
		# array bounds check and g(u) function from Eq. 7.16
		if z′[u] != blank && u != U′-1 && z′[u] != z′[u+2]
		  β[u,t] = logaddexp(β[u,t], ŷ[z′[u+2], t+1] + β[u+2, t+1])
	    end
	  end
    end
  end

  # Loss is taken as the product (sum in log space) of the last two
  # cells in the last column in α
  loss = -1 * logaddexp(α[end,T], α[end-1, T])
  accum = fill(log(typed_zero), size(ŷ))
  grads = fill(log(typed_zero), size(ŷ))
  for t=1:T
    for u=1:U′
      accum[z′[u], t] = logaddexp(accum[z′[u], t], α[u,t] + β[u,t])
    end
    for u=1:size(grads, 1)
      grads[u,t] = exp(ŷ[u, t]) - exp(accum[u, t] - -loss)
    end
  end
  return loss, grads
end

"""
  ctc_loss(ŷ, y)

Computes the connectionist temporal classification loss between `ŷ`
and `y`.

Both `ŷ` and `y` must be classes-by-time matrices, i.e., each row
represents a class and each column represents a time step.
Additionally, the `logsoftmax` function will be applied to `ŷ`, so
it must be the raw activation values from the neural network and
not, for example, the activations after being passed through a
`softmax` activation function.

Used for sequence to sequence classification problems such as
speech recognition and handwriting recognition where the exact
time-alignment of the output (e.g., letters) is not needed to
solve the problem. See [Graves et al. (2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
or [Graves (2012)](https://www.cs.toronto.edu/~graves/preprint.pdf#chapter.7)
for mathematical details.
"""
function ctc_loss(ŷ::Array, y::Array)
  return ctc_(ŷ, y)[1]
end

@adjoint function ctc_(ŷ, y)
  ls, gs = ctc_(ŷ, y)
  return mean(ls), Δ -> (Δ .* gs, Δ)
end
