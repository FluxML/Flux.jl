# GPU impelmentation

# a port of the GPU kernels from Baidu's C++ warp-ctc package
# GitHub: https://github.com/baidu-research/warp-ctc/
# paper: https://arxiv.org/pdf/1512.02595.pdf

using Flux
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
  
  # Fill in first column (time step)
  i = tid
  while i <= last - start
    alpha[start + i] = probs[labels[start + i]]
    i += blockDim().x
  end
  
  sync_threads()
  
  # Fill in coefficients for each time step
  for t=2:T
    startCurCol = (t-1) * S
    startPrevCol = (t-2) * S
    startProbCol = (t-1) * div(length(probs), T)
    
    # Corner-case checking
    if tid == 1 && !(1 < S - 2*(T-t) - 1)
      if start == 0
        alpha[startCurCol + 1] = probs[startProbCol + blankLabel] + alpha[startPrevCol + 1]
      elseif start == 1
        alpha[startCurCol + 1] = alpha[startPrevCol + 1]
      end
    end
    
    sync_threads()
    
    # Fill in coefficients for each label class in the target output sequence;
    # each thread will process the calculations for one class
    idx = tid+1
    while idx <= S
      
      prevSum = log_plus_f(alpha[startPrevCol + idx], alpha[startPrevCol + idx-1])
      
      if labels[idx] != blankLabel && idx != 2 && labels[idx] != labels[idx-2]
        prevSum = log_plus_f(prevSum, alpha[startPrevCol + idx-2])
      end
      
      if idx < S - 2*(T-t) - 1
        alpha[idx + startCurCol] = -Inf32
      else
        alpha[startCurCol + idx] = prevSum + probs[startProbCol + labels[idx]]
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
  
  
  startCurCol = (T-1)*S
  startProbCol = (T-1) * div(length(probs), T)
  
  i = tid
  
  # Calculate coefficients for last column (time step)
  # then determine alpha and beta product
  while i <= last - start + 1

    beta[startCurCol + i + start] = 0
    output[startCurCol + i + start] = beta[startCurCol + i + start] + alphas[startCurCol + i + start]
    i += blockDim().x
  end
  
  sync_threads()
  
  # Fill in `accum` for last column (time step)
  if tid == 1
    startAccCol = startProbCol
    startOutputCol = startCurCol
    
    for i=1:S
      labelIdx = labels[i]
      accum[startAccCol + labelIdx] = log_plus_f(accum[startAccCol + labelIdx], output[startOutputCol + i])
    end
  end
  
  sync_threads()
  
  # Fill in `grad` for last column (time step)
  idx = tid
  while idx <= CUDAnative.div_fast(Float32(length(grad)), Float32(T))
#     
    startProbCol = (T - 1) * div(length(probs), T)
    startOutputCol = (T - 1) * S
    
    s = -Inf32
    for i=1:S
      s = log_plus_f(s, output[startOutputCol + i])
    end
    
    # ∂L/∂a (where a is activation before logsoftmax)
    grad[startProbCol + idx] = CUDAnative.exp(probs[startProbCol + idx]) - CUDAnative.exp(accum[startProbCol + idx] - s)
    idx += blockDim().x
  end
  
  sync_threads()
  
  # Fill in the rest of the coefficients
  t = T-1
  while t >= 1
  
    startCurCol = (t-1)*S
    startNextCol = t*S
    startProbCol = t * div(length(probs), T)

    if t < T
      
      idx = tid
      while idx <= S-1
        
        nextSum = log_plus_f(beta[startNextCol + idx] + probs[startProbCol + labels[idx]],
          beta[startNextCol + idx+1] + probs[startProbCol + labels[idx+1]])
        
        if labels[idx] != blankLabel && idx != S-1 && labels[idx] != labels[idx+2]
          nextSum = log_plus_f(nextSum,
            beta[startNextCol + idx + 2] + probs[startProbCol + labels[idx+2]])
        end
        
        if idx > 2*t
          beta[idx + startCurCol] = -Inf32
        else
          beta[idx + startCurCol] = nextSum
            
        end
        
        idx += blockDim().x
      end
    
      sync_threads()
      
      if tid == 1 && last == S
        beta[startCurCol + S] = beta[startNextCol + S] + probs[startProbCol + blankLabel]
      end
      
      sync_threads()
      
      idx = tid
      while idx <= S
        output[startCurCol + idx] = alphas[idx+startCurCol] + beta[startCurCol + idx]
        idx += blockDim().x
      end
      
      sync_threads()
    end
    
    
    sync_threads()
    
    # Calculate accumulated alpha-beta products for each label class for
    # each time step; used in calculating gradients
    if tid == 1
    
      startAccCol = (t-1) * div(length(accum), T)
      startOutputCol = (t-1) * S
      
      for i=1:S
        labelIdx = labels[i]
        accum[startAccCol + labelIdx] = log_plus_f(accum[startAccCol + labelIdx], output[startOutputCol + i])
      end
    end
    
    sync_threads()
    
    idx = tid
    
    # Calculate gradients
    while idx <= CUDAnative.div_fast(Float32(length(grad)), Float32(T))
#     
      startProbCol = (t - 1) * div(length(probs), T)
      startOutputCol = (t - 1) * S
      
      s = -Inf32
      for i=1:S
        s = log_plus_f(s, output[startOutputCol + i])
      end
      
      # ∂L/∂a (where a is activation before logsoftmax)
      grad[startProbCol + idx] = CUDAnative.exp(probs[startProbCol + idx]) - CUDAnative.exp(accum[startProbCol + idx] - s)
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

ctc(ŷ::CuArrays.CuArray, y::Array) = ctc_(ŷ, y)[1] |> mean

ctc(ŷ::Array, y::CuArrays.CuArray) = ctc_(CuArray(ŷ), y)[1] |> mean

ctc(ŷ::CuArrays.CuArray, y::CuArrays.CuArray) = ctc_(ŷ, y)[1] |> mean

# methods for `ctc_` helper function
ctc_(ŷ::Array, y::CuArrays.CuArray) =  ctc_(CuArray(ŷ), y)

function ctc_(ŷ::CuArrays.CuArray, y)
  
  ŷ = logsoftmax(ŷ)
  
  blank = size(ŷ, 1)
  labels = vec(mapslices(Base.argmax, y, dims=1))
  z = F(labels, blank)
  z′ = [blank]
  for label in z
    push!(z′, label)
    push!(z′, blank)
  end
  T = size(ŷ, 2)
  U′ = 2*length(z) + 1
  alphas = CuArrays.fill(log(zero(ŷ[1])), T * U′)
  betas = copy(alphas)
  output = copy(alphas)
  
  nRepeats = countRepeats(labels)

  # 1 block with `U′` threads
  @cuda blocks=1 threads=U′ computeAlphaKernel(ŷ, length(z), size(ŷ,2), nRepeats, CuArray(z), CuArray(z′), alphas, blank)
  
  grads = CuArrays.fill(log(zero(ŷ[1])), length(ŷ))
  accum = copy(grads)
  
  @cuda blocks=1 threads=U′ computeBetasAndGradKernel(ŷ, length(z), size(ŷ,2), nRepeats, CuArray(z′), alphas, betas, output, accum, grads, blank)
  
  ls = reshape(collect(output), U′, T)
  ls = -1 .* mapslices(logsum, ls, dims=1) |> vec
  
  gs = reshape(grads, size(ŷ,1), size(ŷ,2))
  
  ŷ = alphas = betas = output = accum = grads = nothing
  return ls, gs
end
