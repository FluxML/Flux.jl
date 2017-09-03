call(f, xs...) = f(xs...)

function optimiser(ps, fs...)
  ps = [Param(p) for p in ps]
  fs = map(ps) do p
    os = map(f -> f(p), fs)
    () -> foreach(call, os)
  end
  () -> foreach(call, fs)
end

SGD(ps, η = 1) = optimiser(ps, p -> descent(p, η))
