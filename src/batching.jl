export batch, Batch

# TODO: support the Batch type only
batch(x) = reshape(x, (1,size(x)...))
batch(xs...) = vcat(map(batch, xs)...)

type Batch{T,T′} <: AbstractVector{T}
  data::T′
end

Base.size(b::Batch) = (size(b.data, 1),)

Base.getindex(b::Batch, i)::eltype(b) = slicedim(b.data, 1, i)

Base.setindex!(b::Batch, v, i) = b[i, :] = v

function (::Type{Batch{T}}){T}(xs)
  x = first(xs)
  batch = similar(x, length(xs), size(x)...)
  for i = 1:length(xs)
    @assert size(xs[i]) == size(x)
    batch[i, :] = xs[i]
  end
  return Batch{T,typeof(batch)}(batch)
end

function Batch(xs)
  xs′ = promote(xs...)
  Batch{typeof(xs′[1])}(xs′)
end

@render Juno.Inline b::Batch begin
  Tree(Row(Text("Batch of "), eltype(b),
           Juno.fade("[$(length(b))]")),
           Juno.trim(collect(b)))
end
