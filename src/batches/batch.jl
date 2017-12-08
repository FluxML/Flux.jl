struct Batch{T,A,M}
  data::A
  mask::M
end

Batch{T}(data, mask) where T = Batch{T,typeof(data),typeof(mask)}(data, mask)

Batch(xs) = Batch{typeof(first(xs))}(Flux.batch(xs),trues(length(xs)))
