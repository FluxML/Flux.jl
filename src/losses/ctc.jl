using Flux
using Zygote: @adjoint
using Statistics

# CPU implementation

"""
  logadd(a, b)

Adds log-space `a` and `b` such that the result equals `log(exp(a)+exp(b))`
"""
function logadd(a, b)
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
  logsum(a)

Sums the elements in `a` such that the result equals `log(sum(exp.(a)))`
"""
function logsum(a::AbstractArray)
  local s
  s = a[1]
  for item in a[2:end]
    s = logadd(s, item)
  end
  return s
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
``    end
    prev = curr
  end
  return z
end

"""
  addBlanks(z)

Adds blanks to the start and end of `z`, and between item in `z`
"""
function addBlanks(z, blank)

  z′ = [blank]
  for label in z
    push!(z′, label)
    push!(z′, blank)
  end
  return z′
end

function ctc_(ŷ, y)

  ŷ = logsoftmax(ŷ)
  blank = size(ŷ, 1)

  z = F(Base.argmax.([y[:,i] for i=1:size(y,2)]), blank)
  z′ = addBlanks(z, blank)
  T = size(ŷ, 2)
  U = length(z)
  U′ = length(z′)

  # Calculate α coefficients, from the upper-left, to the bottom-right
  α = zeros(Float64, T, U′)
  for t=1:T
    for u=1:U′
      if t == u == 1
#         α[t,u] = ŷ[t, blank]
        α[t,u] = ŷ[blank, t]
      elseif t == 1 && u == 2
#         α[t,u] = ŷ[t, z′[2]]
        α[t,u] = ŷ[z′[2], t]
      elseif t == 1 && u > 2
        α[t,u] = -Inf
      elseif u < U′ - 2(T - t) - 1
        α[t,u] = -Inf
      else
        idx = u - 2
        idx += z′[u] == blank || (u > 2 && z′[u-2] == z′[u])
        idx = max(1, idx)

        α[t,u] = ŷ[z′[u], t] + logsum(α[t-1, idx:u])
      end
    end
  end

  # Calculate beta coefficients, from the bottom-right, to the upper-left
  β = zeros(Float64, T, U′)
  for i=1:length(β)
    β[i] = -Inf
  end

  # Fill bottom-right corner so bounding errors can be avoided
  # by starting `u` at `U′-1`
  β[T,U′] = 0.0

  for t=T:-1:1
    for u=(U′-1):-1:1
      if t == T && u >= U′ - 1
        β[t,u] = 0.0
      elseif t == T && u < U′ - 1
        continue
      elseif u > 2t || u > U′ + 1
        continue
      else
        idx = u+2
        idx -= z′[u] == blank || (idx < U′ && z′[u+2] == z′[u])
        idx = min(idx, U′)

        v = [β[t+1,i] + ŷ[z′[i], t+1] for i=u:idx]
        β[t, u] = logsum(v)
      end
    end
    if t < T-1
      β[t, U′] = β[t+1, U′] + ŷ[blank, t]
    end
  end

  # Loss at each time t is taken as the sum of the product of the α and β coefficients for
  # all the label classes at time t
  losses = Vector()
  for t=1:T
    v = [α[t,u] + β[t,u] for u in 1:U′]
    push!(losses, -logsum(v))
  end

  # `accum` will hold the sum of the α and β coefficients for
  # each label class at time t; used in calculating gradients
  accum = fill(-Inf, size(ŷ))
  grads = fill(-Inf, size(ŷ))

  for t=1:T
    for u=1:U′
      accum[z′[u], t] = logadd(accum[z′[u], t], α[t,u] + β[t,u])
    end
    for u=1:size(grads, 1)
      grads[u,t] = exp(ŷ[u, t]) - exp(accum[u, t] - -losses[t])
    end
  end

  losses = [x for x in losses]

  return losses, grads
end

"""
  ctc(ŷ, y)

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
function ctc(ŷ::Array, y::Array)
  return ctc_(ŷ, y)[1] |> mean
end

@adjoint function ctc_(ŷ, y)
  ls, gs = ctc_(ŷ, y)
  return mean(ls), Δ -> (Δ .* gs, Δ)
end
