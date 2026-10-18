@testset "Reactant conv layer gradients" begin
    r(dims...) = randn(Float32, dims...)
    # (model, inputs, name, keyword arguments for `test_gradients`)
    test_reactant_gradients([
        (Conv((3, 3), 2 => 4; stride = 2), (r(8, 8, 2, 2),), "Conv stride=2", (;)),
        (Conv((3, 3), 2 => 4; dilation = 2), (r(8, 8, 2, 2),), "Conv dilation=2", (;)),
        (Conv((3, 3), 4 => 4; groups = 2), (r(6, 6, 4, 2),), "Conv groups=2", (;)),
        (Conv((3, 3), 2 => 3; pad = (1, 2, 0, 1)), (r(6, 6, 2, 2),), "Conv pad=(1, 2, 0, 1)", (;)),
        (Conv((3,), 2 => 3), (r(8, 2, 2),), "Conv 1D", (;)),
        (Conv((2, 2, 2), 2 => 3), (r(5, 5, 5, 2, 2),), "Conv 3D", (;)),
        (CrossCor((3, 3), 2 => 3), (r(6, 6, 2, 2),), "CrossCor", (;)),
        (DepthwiseConv((3, 3), 4 => 8), (r(6, 6, 4, 2),), "DepthwiseConv", (;)),
        (ConvTranspose((3, 3), 3 => 2; stride = 2, pad = 1), (r(5, 5, 3, 2),), "ConvTranspose stride=2 pad=1", (;)),
        (ConvTranspose((3, 3), 4 => 4; groups = 2), (r(5, 5, 4, 2),), "ConvTranspose groups=2", (;)),
    ])
end
