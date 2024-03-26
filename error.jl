using Enzyme, Flux
loss(model, x) = sum(model(x))
model = Flux.Bilinear(2 => 3)
x = randn(Float32, 2, 4)
d_model = Flux.fmap(model) do x
    x isa Array ? zero(x) : x
end
Enzyme.autodiff(Reverse, loss, Active, Duplicated(model, d_model), Const(x))
