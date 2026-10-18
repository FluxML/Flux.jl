@testset "Reactant normalisation layer gradients" begin
    r(dims...) = randn(Float32, dims...)
    # (model, inputs, name, keyword arguments for `test_gradients`)
    test_reactant_gradients([
        (Chain(Conv((3, 3), 2 => 3), BatchNorm(3)), (r(6, 6, 2, 4),), "BatchNorm 4D", (;)),
        (Chain(Conv((3, 3), 2 => 3), BatchNorm(3)), (r(6, 6, 2, 4),), "BatchNorm 4D, test mode", (; test_mode = true)),
        (Chain(Conv((3, 3), 2 => 3), InstanceNorm(3)), (r(6, 6, 2, 4),), "InstanceNorm", (;)),
        (Chain(Conv((3, 3), 2 => 4), GroupNorm(4, 2)), (r(6, 6, 2, 4),), "GroupNorm", (;)),
        (LayerNorm((4, 3)), (r(4, 3, 5),), "LayerNorm((4, 3))", (;)),
        (WeightNorm(Dense(3 => 4)), (r(3, 5),), "WeightNorm(Dense)", (;)),
    ])
end
