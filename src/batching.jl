export batch

# TODO: support the Batch type only
batch(x) = reshape(x, (1,size(x)...))
batch(xs...) = vcat(map(batch, xs)...)

type Batch{T,T′} <: AbstractVector{T}
  data::T′
end

Base.size(b::Batch) = (size(b.data, 1),)

Base.getindex(b::Batch, i) = slicedim(b.data, 1, i)::eltype(b)

Base.setindex!(b::Batch, v, i) = b[i, :] = v

function (::Type{Batch{T}}){T}(xs::T...)
  length(xs) == 1 || @assert ==(map(size, xs)...)
  batch = similar(xs[1], length(xs), size(xs[1])...)
  for i = 1:length(xs)
    batch[i, :] = xs[i]
  end
  return Batch{T,typeof(batch)}(batch)
end

function Batch(xs...)
  xs′ = promote(xs...)
  Batch{typeof(xs′[1])}(xs′...)
end

@render Juno.Inline b::Batch begin
  Tree(Row(Text("Batch of "), eltype(b),
           Juno.fade("[$(length(b))]")),
           Juno.trim(collect(b)))
end
