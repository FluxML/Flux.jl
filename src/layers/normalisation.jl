"""
    testmode!(m)
    testmode!(m, false)

Put layers like [`Dropout`](@ref) and `BatchNorm` into testing mode (or back to
training mode with `false`).
"""
function testmode!(m, val::Bool=true)
  prefor(x -> _testmode!(x, val), m)
  return m
end

_testmode!(m, test) = nothing

"""
    Dropout(p)

A Dropout layer. For each input, either sets that input to `0` (with probability
`p`) or scales it by `1/(1-p)`. This is used as a regularisation, i.e. it
reduces overfitting during training.

Does nothing to the input once in [`testmode!`](@ref).
"""
mutable struct Dropout{F}
  p::F
  active::Bool
end

function Dropout(p)
  @assert 0 ≤ p ≤ 1
  Dropout{typeof(p)}(p, true)
end

function (a::Dropout)(x)
  a.active || return x
  y = similar(x)
  rand!(y)
  q = 1 - a.p
  @inbounds for i=1:length(y)
    y[i] = y[i] > a.p ? 1 / q : 0
  end
  return y .* x
end

_testmode!(a::Dropout, test) = (a.active = !test)
