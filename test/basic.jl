xs = randn(10)' # TODO: batching semantics

d = Affine(10, 20)

@test d(xs) == xs*d.W.x + d.b.x
