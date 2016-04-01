export Dense

type Dense <: Model
  W::Matrix{Float32}
  b::Vector{Float32}
  ∇W::Matrix{Float32}
  ∇b::Vector{Float32}

  in::Vector{Float32}
  out::Vector{Float32}
  ∇in::Vector{Float32}
end

Dense(in::Integer, out::Integer) =
  Dense(randn(out, in), randn(out),
        zeros(out, in), zeros(out),
        zeros(in), zeros(out), zeros(in))

Dense(out::Integer) = Init(in -> Dense(in[1], out))

function (l::Dense)(x)
  l.in = x
  A_mul_B!(l.out, l.W, x)
  map!(+, l.out, l.out, l.b)
end

function back!(l::Dense, ∇)
  map!(+, l.∇b, l.∇b, ∇)
  # l.∇W += ∇ * l.in'
  BLAS.gemm!('N', 'T', eltype(∇)(1), ∇, l.in, eltype(∇)(1), l.∇W)
  At_mul_B!(l.∇in, l.W, ∇)
end

function update!(l::Dense, η)
  map!((x, ∇x) -> x - η*∇x, l.W, l.W, l.∇W)
  map!((x, ∇x) -> x - η*∇x, l.b, l.b, l.∇b)
  fill!(l.∇W, 0)
  fill!(l.∇b, 0)
end

shape(d::Dense) = size(d.b)
