dropout_mask(rng::CUDA.RNG, x::CuArray, p; kwargs...) = _dropout_mask(rng, x, p; kwargs...)
dropout_mask(rng, x::CuArray, p; kwargs...) =
  throw(ArgumentError("x isa CuArray, but rng isa $(typeof(rng)). dropout_mask only support CUDA.RNG for CuArrays."))
