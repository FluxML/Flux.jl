# Gradient parity against Zygote for convolution, pooling and upsampling layers and options not in
# `TEST_MODELS`. Each case chains several layers, as every case costs two compilations.

@testset "Reactant conv layer gradients" begin
    r(dims...) = randn(Float32, dims...)
    # (model, inputs, name, keyword arguments for `test_gradients`)
    test_reactant_gradients([
        (Chain(Conv((3, 3), 2 => 4; stride = 2, pad = (1, 2, 0, 1)), Conv((3, 3), 4 => 4; dilation = 2, groups = 2)),
         (r(12, 12, 2, 2),), "Conv stride, asymmetric pad, dilation, groups", (;)),
        (Chain(Conv((3,), 2 => 3), CrossCor((3,), 3 => 3)), (r(8, 2, 2),), "Conv, CrossCor 1D", (;)),
        (Conv((2, 2, 2), 2 => 3), (r(5, 5, 5, 2, 2),), "Conv 3D", (;)),
        (Chain(DepthwiseConv((3, 3), 4 => 8), ConvTranspose((3, 3), 8 => 4; stride = 2, pad = 1, groups = 2)),
         (r(6, 6, 4, 2),), "DepthwiseConv, ConvTranspose stride, pad, groups", (;)),

        (Chain(Conv((3, 3), 2 => 3), MaxPool((2, 2)), MeanPool((2, 2))), (r(10, 10, 2, 2),), "MaxPool, MeanPool", (;)),
        (Chain(Conv((3, 3), 2 => 3), Parallel(+, Chain(AdaptiveMaxPool((2, 2)), GlobalMeanPool()),
                                                Chain(AdaptiveMeanPool((2, 2)), GlobalMaxPool()))),
         (r(8, 8, 2, 2),), "Adaptive and global pooling", (;)),
        (Chain(Conv((3, 3), 2 => 4), Upsample(:nearest; scale = 2), Upsample(:bilinear; scale = 2), PixelShuffle(2)),
         (r(6, 6, 2, 2),), "Upsample nearest and bilinear, PixelShuffle", (;)),
    ])
end
