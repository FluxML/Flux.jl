# GPU impelmentation

# a port of the GPU kernels from Baidu's C++ warp-ctc package
# GitHub: https://github.com/baidu-research/warp-ctc/
# paper: https://arxiv.org/pdf/1512.02595.pdf

using Flux
using Flux.Tracker: @grad
using Statistics
using CuArrays
using CUDAnative

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

# @require CUDAnative begin
# @require CuArrays begin
# using CUDAnative, CuArrays
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
  return ctc_(ŷ::CuArrays.CuArray, y)[1]
end

function ctc_(ŷ::CuArrays.CuArray, y)

  ŷ = logsoftmax(ŷ)
  floatType = typeof(ŷ[1]) 
  
  blank = size(ŷ, 1)
  labels = vec(mapslices(argmax, y, dims=1))
  z = F(labels, blank)
  z′ = [blank]
  for label in z
    push!(z′, label)
    push!(z′, blank)
  end
  T = size(ŷ, 2)
  U′ = 2*length(z) + 1
  alphas = zeros(floatType, T * U′) |> CuArray |> x -> log.(x)
  betas = copy(alphas)
  output = copy(alphas)
  
  nRepeats = countRepeats(labels)

  # 1 block with `U′` threads
  @cuda blocks=1 threads=U′ computeAlphaKernel(ŷ, length(z), size(ŷ,2), nRepeats, CuArray(z), CuArray(z′), alphas, blank)
  grads = zeros(floatType, length(ŷ)) |> CuArray |> x -> log.(x)
  accum = copy(grads)
  
  @cuda blocks=1 threads=U′ computeBetasAndGradKernel(ŷ, length(z), size(ŷ,2), nRepeats, CuArray(z′), alphas, betas, output, accum, grads, blank)
  
  ls = Array(reshape(Array(output), U′, T)')
  ls = -1 .* mapslices(logsum, ls, dims=2) |> vec
  
  gs = reshape(grads, size(ŷ,1), size(ŷ,2)
  
  ŷ = alphas = betas = output = accum = grads = nothing
  return ls, gs
end
# end # end of `@require CuArrays`
# end # end of `@require CUDAnative`
