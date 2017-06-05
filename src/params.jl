"""
A `Param` object stores a parameter array along with its gradient.
When converting to backends like TensorFlow, identical `Param`s will
result in identical variable objects.
"""
struct Param{T}
  x::T
  Δx::T
end

"""
    param(x::T) => ::Param{T}

Convenience method for creating a `Param` object for a given array.
"""
param(x) = Param(x, zero(x))

state(p::Param) = p.x

"""
    update!(p::Param)

Apply the accumulated updates to the value of the parameter.
"""
function update!(p::Param, η)
  p.x .-= p.Δx .* η
  p.Δx[:] = 0
  return p
end

state(x) = x

Base.size(p::Param) = size(p.x)
Base.size(p::Param, n) = size(p.x, n)

function Base.show(io::IO, p::Param)
  print(io, "Param", size(p.x))
end

Base.copy!(xs, p::Param) = copy!(xs, p.x)
Base.copy!(p::Param, xs) = copy!(p.x, xs)
