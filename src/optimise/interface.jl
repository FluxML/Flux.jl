call(f, xs...) = f(xs...)

function optimiser(m, fs...)
  ps = Param.(params(m))
  fs = map(ps) do p
    os = map(f -> f(p), fs)
    () -> foreach(call, os)
  end
  () -> foreach(call, fs)
end

SGD(m, η = 1) = optimiser(m, p -> descent(p, 0.1))
