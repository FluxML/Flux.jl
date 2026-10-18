@testset "Reactant upsampling layer gradients" begin
    r(dims...) = randn(Float32, dims...)
    # (model, inputs, name, keyword arguments for `test_gradients`)
    test_reactant_gradients([
        (Chain(Conv((3, 3), 2 => 3), Upsample(:nearest; scale = 2)), (r(6, 6, 2, 2),), "Upsample nearest", (;)),
        (Chain(Conv((3, 3), 2 => 3), Upsample(:bilinear; scale = 2)), (r(6, 6, 2, 2),), "Upsample bilinear", (;)),
        (Chain(Conv((3, 3), 2 => 4), PixelShuffle(2)), (r(6, 6, 2, 2),), "PixelShuffle", (;)),
    ])
end
