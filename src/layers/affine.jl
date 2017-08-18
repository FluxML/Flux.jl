@net type Affine
  W
  b
  x -> x*W .+ b
end

Affine(in::Integer, out::Integer; init = initn) =
  Affine(init(in, out), init(1, out))

function back!(m::Affine, Δ, x)
  W, b = m.W, m.b
  W.Δx[:] = x' * Δ
  b.Δx[:] = sum(Δ, 1)
  Δ * W.x'
end

function update!(m::Affine, η)
  update!(m.W, η)
  update!(m.b, η)
  m
end
