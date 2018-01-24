function gradient(f, xs::AbstractArray...)
  xs = param.(xs)
  back!(f(xs...))
  grad.(xs)
end

function ngradient(f, xs::AbstractArray...)
  grads = zeros.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

function gradient_cat(f, dim, xs::AbstractArray...)
  xs = param.(xs)
  back!(f(dim, xs...))
  grad.(xs)
end

function ngradient_cat(f, dim, xs::AbstractArray...)
  grads = zeros.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(dim, xs...)
    x[i] = tmp + δ/2
    y2 = f(dim, xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

gradcheck(f, xs...) = all(isapprox.(ngradient(f, xs...), gradient(f, xs...), rtol = 1e-5))
gradcheck_cat(f, dim, xs...) = all(isapprox.(ngradient_cat(f, dim, xs...), gradient_cat(f, dim, xs...), rtol = 1e-5))
