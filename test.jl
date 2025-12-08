using CUDA
using Flux
using Enzyme

model = Dense(2 => 2) |> gpu
dmodel = Duplicated(model, Enzyme.make_zero(model))
ad = Enzyme.set_runtime_activity(Reverse)
f(m, x) = sum(m(x))
x = rand(Float32, 2) |> gpu
Enzyme.autodiff(ad, Const(f), Active, dmodel, Const(x))



using NNlib
using Enzyme: Const, Duplicated
using EnzymeTestUtils

src = Float64[3, 4, 5, 6, 7]
idx = [
    1 2 3 4;
    4 2 1 3;
    3 5 5 3]
dst = gather(src, idx)

EnzymeTestUtils.test_reverse(gather!, Tret, (dst, Const), (src, Duplicated), (idx, Const))