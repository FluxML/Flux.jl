# This can be made an error in Flux v0.13, for now just a warning
function _check_sizes(ŷ::AbstractArray, y::AbstractArray)
  for d in 1:max(ndims(ŷ), ndims(y))
    if size(ŷ,d) != size(y,d)
      @warn "Size mismatch in loss function! In future this will be an error. In Flux <= 0.12 broadcasting accepts this, but may not give sensible results" summary(ŷ) summary(y) maxlog=3 _id=hash(size(y))
    end
  end
end
_check_sizes(ŷ, y) = nothing  # pass-through, for constant label e.g. y = 1

Zygote.@nograd _check_sizes
