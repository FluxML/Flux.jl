_loadleaf(x) = isleaf(x)
for T in [:Dense, :Diagonal, :Bilinear, :Embedding,
          :Conv, :ConvTranspose, :DepthwiseConv, :CrossCor]
  @eval _loadleaf(::$T) = true
end

loadto!(x, x̄) = x
loadto!(x::AbstractArray, x̄::AbstractArray) = copyto!(x, x̄)
for T in [:Dense, :Bilinear, :Conv, :ConvTranspose, :DepthwiseConv, :CrossCor]
  @eval begin
    function loadto!(m::$T, m̄::$T)
      if (size(m.weight) != size(m̄.weight)) || (size(m.bias) != size(m̄.bias))
        throw(DimensionMismatch("Tried to load $m̄ into $m but the parameter sizes do not match."))
      else
        return fmap(loadto!, m, m̄)
      end
    end
    loadto!(m::$T, m̄) = throw(ArgumentError("Tried to load $m̄ into $m."))
  end
end
function loadto!(m::Diagonal, m̄::Diagonal)
  if (size(m.α) != size(m̄.α)) || (size(m.β) != size(m̄.β))
    throw(DimensionMismatch("Tried to load $m̄ into $m but the parameter sizes do not match."))
  else
    return fmap(loadto!, m, m̄)
  end
end
loadto!(m::Diagonal, m̄) = throw(ArgumentError("Tried to load $m̄ into $m."))
function loadto!(m::Embedding, m̄::Embedding)
  if size(m.weight) != size(m̄.weight)
    throw(DimensionMismatch("Tried to load $m̄ into $m but the parameter sizes do not match."))
  else
    return fmap(loadto!, m, m̄)
  end
end
loadto!(m::Embedding, m̄) = throw(ArgumentError("Tried to load $m̄ into $m."))

function loadmodel!(m, xs::Params)
  for (p, x) in zip(params(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(p, x)
  end
end
loadmodel!(m, xs::AbstractVector) = loadmodel!(m, params(xs))
loadmodel!(m, m̄) = fmap(loadto!, m, m̄; exclude = _loadleaf)
