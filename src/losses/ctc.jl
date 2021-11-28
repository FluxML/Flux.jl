using Flux
using Zygote: @adjoint
using Statistics
using NNlib

## CPU implementation

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
    return a + log(1 + exp(b - a))
end

"""
  add_blanks(z)

Adds blanks to the start and end of `z`, and between items in `z`
"""
function add_blanks(z, blank)
    z′ = fill(blank, 2 * length(z) + 1)
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
    α[1, 1] = ŷ[blank, 1]
    α[2, 1] = ŷ[z′[2], 1]
    for t in 2:T
        bound = max(1, U′ - 2(T - t) - 1)
        for u in bound:U′
            if u == 1
                α[u, t] = α[u, t - 1]
            else
                α[u, t] = logaddexp(α[u, t - 1], α[u - 1, t - 1])

                # array bounds check and f(u) function from Eq. 7.9
                if u > 2 && !(z′[u] == blank || z′[u - 2] == z′[u])
                    α[u, t] = logaddexp(α[u, t], α[u - 2, t - 1])
                end
            end
            α[u, t] += ŷ[z′[u], t]
        end
    end
    return (
        loss = -1 * logaddexp(α[end, T], α[end - 1, T]),
        alpha = α,
        zprime = z′,
        logsoftyhat = ŷ,
    )
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
    β[U′ - 1, T] = typed_zero

    # start at T-1 so that β(T, u) = log(0) for all u < U′ - 1
    for t in (T - 1):-1:1
        bound = min(U′, 2t)
        for u in bound:-1:1
            if u == U′
                β[u, t] = ŷ[z′[u], t + 1] + β[u, t + 1]
            else
                β[u, t] = logaddexp(
                    ŷ[z′[u], t + 1] + β[u, t + 1], ŷ[z′[u + 1], t + 1] + β[u + 1, t + 1]
                )

                # array bounds check and g(u) function from Eq. 7.16
                if u + 2 <= U′ && z′[u] != blank && z′[u] != z′[u + 2]
                    β[u, t] = logaddexp(β[u, t], ŷ[z′[u + 2], t + 1] + β[u + 2, t + 1])
                end
            end
        end
    end

    # Accumulate alpha-beta products for each category,
    # then calculate gradients
    accum = fill(log(typed_zero), size(ŷ))
    for t in 1:T
        for u in 1:U′
            accum[z′[u], t] = logaddexp(accum[z′[u], t], α[u, t] + β[u, t])
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

## GPU implementation

# a port of the GPU kernels from Baidu's C++ warp-ctc package,
# which itself is Copyright 2015-2016 Baidu USA LLC
# and available under the Apache 2.0 license
#
# Apache 2.0 license: https://www.apache.org/licenses/LICENSE-2.0
# GitHub: https://github.com/baidu-research/warp-ctc/
# paper: https://arxiv.org/pdf/1512.02595.pdf

const MAX_THREADS = 256

function log_plus_f(p1, p2)
    isinf(p1) && return p2
    isinf(p2) && return p1
    if p1 < p2
        p1, p2 = p2, p1
    end
    return p1 + log(1 + exp(p2 - p1))
end

function count_repeats(A)
    repeats = 0
    for (i, elem) in enumerate(A)
        if i > 1 && A[i] == A[i - 1]
            repeats += 1
        end
    end
    return repeats
end

function compute_alpha_kernel(
    probs,
    labelSize,
    uttLength,
    repeats,
    labelsWithoutBlanks,
    labelsWithBlanks,
    alpha,
    blankLabel,
)
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
        alpha[start + i, 1] = probs[labels[start + i], 1]
        i += blockDim().x
    end
    sync_threads()

    # Fill in coefficients for each time step
    for t in 2:T
        # Corner-case checking
        if tid == 1 && !(1 < S - 2 * (T - t) - 1)
            if start == 0
                alpha[1, t] = alpha[1, t - 1] + probs[blankLabel, t]
            elseif start == 1
                alpha[1, t] = alpha[1, t - 1]
            end
        end
        sync_threads()

        # Fill in coefficients for each label class in the target output sequence;
        # each thread will process the calculations for one class
        idx = tid + 1
        while idx <= S
            prevSum = log_plus_f(alpha[idx, t - 1], alpha[idx - 1, t - 1])
            if labels[idx] != blankLabel && idx != 2 && labels[idx] != labels[idx - 2]
                prevSum = log_plus_f(prevSum, alpha[idx - 2, t - 1])
            end
            if idx < S - 2 * (T - t) - 1
                alpha[idx, t] = -Inf32
            else
                alpha[idx, t] = prevSum + probs[labels[idx], t]
            end
            idx += blockDim().x
        end
        sync_threads()
    end
    return nothing
end

function compute_beta_and_grad_kernel(
    probs,
    labelSize,
    uttLength,
    repeatsInLabel,
    labelsWithBlanks,
    alphas,
    beta,
    output,
    accum,
    grad,
    blankLabel,
    loss,
)
    tid = threadIdx().x
    L = labelSize
    T = uttLength
    S = 2 * L + 1
    repeats = repeatsInLabel
    labels = labelsWithBlanks

    if (L + repeats) > T
        return nothing
    end

    # Corner-case checking
    start = S > 1 ? S - 2 : 0
    last = L + repeats < T ? S : S - 1
    sync_threads()
    i = tid

    # Calculate coefficients for last column (time step)
    # then determine alpha and beta product
    while i <= last - start
        beta[i + start, T] = 0
        output[i + start, T] = beta[i + start, T] + alphas[i + start, T]
        i += blockDim().x
    end
    sync_threads()

    # Fill in `accum` for last column (time step)
    if tid == 1
        for i in 1:S
            labelIdx = labels[i]
            accum[labelIdx, T] = log_plus_f(accum[labelIdx, T], output[i, T])
        end
    end
    sync_threads()

    # Fill in `grad` for last column (time step)
    idx = tid
    while idx <= size(grad, 1)
        s = -Inf32
        for i in 1:S
            s = log_plus_f(s, output[i, T])
        end

        # ∂L/∂a (where a is activation before logsoftmax)
        grad[idx, T] = exp(probs[idx, T]) - exp(accum[idx, T] - s)
        idx += blockDim().x
    end
    sync_threads()

    # Fill in the rest of the coefficients
    t = T - 1
    while t >= 1
        if t < T
            idx = tid
            while idx <= S
                nextSum = probs[labels[idx], t + 1] + beta[idx, t + 1]
                if idx < S
                    nextSum = log_plus_f(
                        nextSum, probs[labels[idx + 1], t + 1] + beta[idx + 1, t + 1]
                    )
                end
                if labels[idx] != blankLabel &&
                    idx != S - 1 &&
                    labels[idx] != labels[idx + 2]
                    nextSum = log_plus_f(
                        nextSum, probs[labels[idx + 2], t + 1] + beta[idx + 2, t + 1]
                    )
                end
                if idx > 2 * t
                    beta[idx, t] = -Inf32
                else
                    beta[idx, t] = nextSum
                end
                idx += blockDim().x
            end
            sync_threads()
            idx = tid
            while idx <= S
                output[idx, t] = alphas[idx, t] + beta[idx, t]
                idx += blockDim().x
            end
            sync_threads()
        end
        sync_threads()

        # Calculate accumulated alpha-beta products for each label class for
        # each time step; used in calculating gradients
        if tid == 1
            for i in 1:S
                labelIdx = labels[i]
                accum[labelIdx, t] = log_plus_f(accum[labelIdx, t], output[i, t])
            end
        end
        sync_threads()
        idx = tid

        # Calculate gradients
        while idx <= size(grad, 1)

            # ∂L/∂a (where a is activation before logsoftmax)
            grad[idx, t] = exp(probs[idx, t]) - exp(accum[idx, t] + loss)
            idx += blockDim().x
        end
        sync_threads()
        t -= 1
        sync_threads()
    end
    return nothing
end

function ctc_alpha(ŷ::CuArray, y)
    ŷ = logsoftmax(ŷ)
    blank = size(ŷ, 1)
    ycu = cu(y)
    z′ = CUDA.fill(blank, 2 * length(y) + 1)
    z′[eachindex(y) .* 2] .= ycu
    T = size(ŷ, 2)
    U′ = 2 * length(y) + 1
    alphas = CUDA.fill(log(zero(eltype(ŷ))), U′, T)
    nRepeats = count_repeats(cpu(y))
    nThreads = min(U′, MAX_THREADS)
    @cuda blocks = 1 threads = nThreads compute_alpha_kernel(
        ŷ, length(y), T, nRepeats, ycu, z′, alphas, blank
    )
    return (
        loss = -1 * logsumexp(alphas[(end - 1):end]),
        alpha = alphas,
        z′ = z′,
        yhat = ŷ,
        nRepeats = nRepeats,
    )
end

ctc_loss(ŷ::CuArray, y) = ctc_alpha(ŷ::CuArray, y).loss

function ∇ctc_loss(ŷ::CuArray, y, out)
    loss, alphas, z′, ŷ, nRepeats = out
    U′, T = size(alphas)
    blank = size(ŷ, 1)
    typed_zero = zero(eltype(ŷ))
    betas = CUDA.fill(log(typed_zero), U′, T)
    output = CUDA.fill(log(typed_zero), U′, T)
    nThreads = min(U′, MAX_THREADS)
    grads = CUDA.fill(log(typed_zero), size(ŷ))
    accum = CUDA.fill(log(typed_zero), size(ŷ))
    @cuda blocks = 1 threads = nThreads compute_beta_and_grad_kernel(
        ŷ,
        length(y),
        T,
        nRepeats,
        CuArray(z′),
        alphas,
        betas,
        output,
        accum,
        grads,
        blank,
        loss,
    )
    return grads
end
