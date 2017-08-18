@net type Affine
  W
  b
  x -> x*W .+ b
end

Affine(in::Integer, out::Integer; init = initn) =
  Affine(init(in, out), init(1, out))
