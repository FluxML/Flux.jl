# using CUDAnative, CUDAdrv
using Flux
using Flux.Tracker: @grad
using Requires

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
function logsum(a)
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

function ctc(ŷ::Array, y)

  ŷ = logsoftmax(ŷ)
  blank = size(ŷ, 1)
  
  z = F(indmax.([y[i,:] for i=1:size(y,1)]), blank)
  z′ = addBlanks(z, blank)
  T = size(ŷ, 2)
  U = length(z)
  U′ = length(z′)

  # Calculate α coefficients, from the upper-left, to the bottom-right
  α = Array{Float64}(T, U′)
  for t=1:T
    for u=1:U′
      if t == u == 1
        α[t,u] = ŷ[t, blank]
      elseif t == 1 && u == 2
        α[t,u] = ŷ[t, z′[2]]
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
  β = Array{Float64}(T, U′)
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
  accum = reshape([-Inf for x=1:length(ŷ )], size(ŷ ))
  grads = reshape([-Inf for x=1:length(ŷ )], size(ŷ ))
  
  for t=1:T
    for u=1:U′
      accum[z′[u], t] = logadd(accum[z′[u], t], α[t,u] + β[t,u])
    end
    for u=1:size(grads, 1)
      grads[u,t] = exp(ŷ[u, t]) - exp(accum[u, t] - -losses[t])
    end
  end

  return mean(losses), grads
end

# GPU impelmentation

# a port of the GPU kernels from Baidu's C++ warp-ctc package
# GitHub: https://github.com/baidu-research/warp-ctc/
# paper: https://arxiv.org/pdf/1512.02595.pdf

function log_plus_f(p1, p2)
  
  isinf(p1) && return p2
  isinf(p2) && return p1

  if p1 < p2
    p1, p2 = p2, p1
  end

  return p1 + CUDAnative.log(1+CUDAnative.exp(p2 - p1))
end

function countRepeats(A)
  repeats = 0
  for (i,elem) in enumerate(A)
    if i > 1 && A[i] == A[i-1]
      repeats += 1
    end
  end
  return repeats
end

@require CUDAnative begin
function computeAlphaKernel(probs, labelSize, uttLength, repeats, labelsWithoutBlanks, labelsWithBlanks, alpha, blankLabel)

  tid = threadIdx().x
  L = labelSize
  T = uttLength
  S = length(labelsWithBlanks)
  
  if L + repeats > T
    return nothing
  end
  
  labels = labelsWithBlanks

  # Corner-case checking
  start = (L + repeats <= T) ? 0 : 1
  last = S > 1 ? 2 : 1
  
  # Fill in first row
  i = tid
  while i <= last - start
    alpha[start + i] = probs[labels[start + i]]
    i += blockDim().x
  end
  
  sync_threads()
  
  # Fill in coefficients for each time step
  for t=2:T
    startCurRow = (t-1) * S
    startPrevRow = (t-2) * S
    startProbCol = (t-1) * div(length(probs), T)
    
    # Corner-case checking
    if tid == 1 && !(1 < S - 2*(T-t) - 1)
      if start == 0
        alpha[startCurRow + 1] = probs[startProbCol + blankLabel] + alpha[startPrevRow + 1]
      elseif start == 1
        alpha[startCurRow + 1] = alpha[startPrevRow + 1]
      end
    end
    
    sync_threads()
    
    # Fill in coefficients for each label class in the target output sequence;
    # each thread will process the calculations for one class
    idx = tid+1
    while idx <= S
      
      prevSum = log_plus_f(alpha[startPrevRow + idx], alpha[startPrevRow + idx-1])
      
      if labels[idx] != blankLabel && idx != 2 && labels[idx] != labels[idx-2]
        prevSum = log_plus_f(prevSum, alpha[startPrevRow + idx-2])
      end
      
      if idx < S - 2*(T-t) - 1
        alpha[idx + startCurRow] = -Inf32
      else
        alpha[startCurRow + idx] = prevSum + probs[startProbCol + labels[idx]]
      end
    
      idx += blockDim().x
    end
    
    sync_threads()
  end
  return nothing
end

function computeBetasAndGradKernel(probs, labelSize, uttLength,
                  repeatsInLabel, labelsWithBlanks,
                  alphas, beta, output, accum,
                  grad, blankLabel)
  
  tid = threadIdx().x
  L = labelSize
  T = uttLength
  S = 2*L + 1
  repeats = repeatsInLabel
  
  labels = labelsWithBlanks
  
  if (L+repeats) > T
    return nothing
  end
  
  # Corner-case checking
  start = S > 1 ? S-2 : 0
  last = L + repeats < T ? S : S-1
  
  sync_threads()
  
  
  startCurRow = (T-1)*S
  startProbCol = (T-1) * div(length(probs), T)
  
  i = tid
  
  # Calculate coefficients for last row, then determine alpha and beta product
  while i <= last - start + 1

    beta[startCurRow + i + start] = 0
    output[startCurRow + i + start] = beta[startCurRow + i + start] + alphas[startCurRow + i + start]
    i += blockDim().x
  end
  
  sync_threads()
  
  # Fill in `accum` for last row
  if tid == 1
    startAccRow = startProbCol
    startOutputRow = startCurRow
    
    for i=1:S
      labelIdx = labels[i]
      accum[startAccRow + labelIdx] = log_plus_f(accum[startAccRow + labelIdx], output[startOutputRow + i])
    end
  end
  
  sync_threads()
  
  # Fill in `grad` for last row
  idx = tid
  while idx <= div(length(grad), T)
#     
    startProbRow = (T - 1) * div(length(probs), T)
    startOutputRow = (T - 1) * S
    
    s = -Inf32
    for i=1:S
      s = log_plus_f(s, output[startOutputRow + i])
    end
    
    # ∂L/∂a (where a is activation before logsoftmax)
    grad[startProbRow + idx] = CUDAnative.exp(probs[startProbRow + idx]) - CUDAnative.exp(accum[startProbRow + idx] - s)
    idx += blockDim().x
  end
  
  sync_threads()
  
  # Fill in the rest of the coefficients
  t = T-1
  while t >= 1
  
    startCurRow = (t-1)*S
    startNextRow = t*S
    startProbCol = t * div(length(probs), T)

    if t < T
      
      idx = tid
      while idx <= S-1
        
        nextSum = log_plus_f(beta[startNextRow + idx] + probs[startProbCol + labels[idx]],
          beta[startNextRow + idx+1] + probs[startProbCol + labels[idx+1]])
        
        if labels[idx] != blankLabel && idx != S-1 && labels[idx] != labels[idx+2]
          nextSum = log_plus_f(nextSum,
            beta[startNextRow + idx + 2] + probs[startProbCol + labels[idx+2]])
        end
        
        if idx > 2*t
          beta[idx + startCurRow] = -Inf32
        else
          beta[idx + startCurRow] = nextSum
            
        end
        
        idx += blockDim().x
      end
    
      sync_threads()
      
      if tid == 1 && last == S
        beta[startCurRow + S] = beta[startNextRow + S] + probs[startProbCol + blankLabel]
      end
      
      sync_threads()
      
      idx = tid
      while idx <= S
        output[startCurRow + idx] = alphas[idx+startCurRow] + beta[startCurRow + idx]
        idx += blockDim().x
      end
      
      sync_threads()
    end
    
    
    sync_threads()
    
    # Calculate accumulated alpha-beta products for each label class for
    # each time step; used in calculating gradients
    if tid == 1
    
      startAccRow = (t-1) * div(length(accum), T)
      startOutputRow = (t-1) * S
      
      for i=1:S
        labelIdx = labels[i]
        accum[startAccRow + labelIdx] = log_plus_f(accum[startAccRow + labelIdx], output[startOutputRow + i])
      end
    end
    
    sync_threads()
    
    idx = tid
    
    # Calculate gradients
    while idx <= div(length(grad), T)
#     
      startProbRow = (t - 1) * div(length(probs), T)
      startOutputRow = (t - 1) * S
      
      s = -Inf32
      for i=1:S
        s = log_plus_f(s, output[startOutputRow + i])
      end
      
      # ∂L/∂a (where a is activation before logsoftmax)
      grad[startProbRow + idx] = CUDAnative.exp(probs[startProbRow + idx]) - CUDAnative.exp(accum[startProbRow + idx] - s)
      idx += blockDim().x
    end
    
    sync_threads()
    
    t -= 1
    sync_threads()
    # because of course, it wouldn't work without this earlier return statement
    # otherwise, some of the gradient values become 0
    t == 0 && return
  end

  return nothing
end

function ctc(ŷ::CuArrays.CuArray, y)

  ŷ = logsoftmax(ŷ)
  
  blank = Int32(size(ŷ, 1))
  labels = indmax.([y[i,:] for i=1:size(y,1)])
  z = F(labels, blank)
  z′ = [blank]
  for label in z
    push!(z′, label)
    push!(z′, blank)
  end
  T = size(ŷ, 2)
  U′ = 2*length(z) + 1
  alphas = gpu([-Inf32 for x in 1:(size(ŷ,2) * U′)])
  betas = gpu([-Inf32 for x in 1:(size(ŷ,2) * U′)])
  
  nRepeats = countRepeats(labels)

#   @cuda (1, U′) computeAlphaKernel(ŷ, length(z), size(ŷ,2), nRepeats, gpu(z), CUDAdrv.CuArray(z′), alphas, blank)
  # Julia 0.7 and updated CUDAnative function call
  @cuda threads=U′ computeAlphaKernel(ŷ, length(z), size(ŷ,2), nRepeats, gpu(z), CUDAdrv.CuArray(z′), alphas, blank)
  grads = gpu([-Inf32 for x in 1:length(ŷ)])
  output = gpu([-Inf32 for x in 1:(size(ŷ,2) * U′)])
  accum = gpu([-Inf32 for x in 1:length(ŷ)])
  
#   @cuda (1, U′) computeBetasAndGradKernel(ŷ, length(z), size(ŷ,2), nRepeats, CUDAdrv.CuArray(z′), alphas, betas, output, accum, grads, blank)
  # Julia 0.7 and updated CUDAnative function call
  @cuda threads=U′ computeBetasAndGradKernel(ŷ, length(z), size(ŷ,2), nRepeats, CUDAdrv.CuArray(z′), alphas, betas, output, accum, grads, blank)
  
  ls = Array(reshape(Array(output), U′, T)')
  ls = -1 .* mapslices(logsum, ls, 2)
  gs = reshape(Array(grads), size(ŷ,1), size(ŷ,2))
  
  ŷ = alphas = betas = output = accum = grads = nothing
  
  return mean(ls), gs
end
end # end of requries

ctc(ŷ::TrackedArray, y::AbstractArray) = Flux.Tracker.track(ctc, ŷ, y)

@grad function ctc(ŷ, y)
  ls, gs = ctc(Flux.Tracker.data(ŷ), Flux.Tracker.data(y))
  return ls, Δ -> (Δ .* gpu(gs), Δ)
end
