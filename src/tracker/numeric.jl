function gradient(f, xs::AbstractArray...)
  xs = track.(xs)
  back!(f(xs...), [1])
  grad.(xs)
end

function ngradient(f, xs::AbstractArray...)
  y = f(xs...)
  grads = zeros.(xs)
  for (x, Δ) in zip(xs, grads)
    for i in 1:length(x)
      δ = sqrt(eps())
      tmp, x[i] = x[i], x[i]+δ
      y′ = f(xs...)
      x[i] = tmp
      Δ[i] = (y′-y)/δ
    end
  end
  return grads
end

gradcheck(f, xs...) = all(isapprox.(ngradient(f, xs...), gradient(f, xs...), rtol = 1e-6))
