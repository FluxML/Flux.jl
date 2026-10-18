@testset "Reactant pooling layer gradients" begin
    r(dims...) = randn(Float32, dims...)
    # (model, inputs, name, keyword arguments for `test_gradients`)
    test_reactant_gradients([
        (Chain(Conv((3, 3), 2 => 3), pool), (r(8, 8, 2, 2),), "Chain(Conv, $pool)", (;))
        for pool in (MaxPool((2, 2)), MeanPool((2, 2)), AdaptiveMaxPool((2, 2)),
                     AdaptiveMeanPool((2, 2)), GlobalMaxPool(), GlobalMeanPool())
    ])
end
