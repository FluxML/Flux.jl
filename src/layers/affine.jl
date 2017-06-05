@net type Affine
  W
  b
  x -> x*W .+ b
end

Affine(in::Integer, out::Integer; init = initn) =
  Affine(init(in, out), init(1, out))

inferred(::Type{Affine}, in::Tuple{Dims{2}}, out::Integer) =
  Affine(in[1][2], out)
