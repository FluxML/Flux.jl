using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    x = rand(Float32, 4, 3)
    xc = rand(Float32, 8, 8, 3, 3)
    y = onehotbatch([1, 2, 1], 1:2)
    yf = rand(Float32, 2, 3)
    @compile_workload begin
        mlp = Chain(Dense(4 => 8, relu), Dropout(0.1), Dense(8 => 8, tanh), BatchNorm(8),
                    Dense(8 => 2))
        cnn = Chain(Conv((3, 3), 3 => 4, relu; pad = 1), BatchNorm(4), MaxPool((2, 2)),
                    Conv((3, 3), 4 => 4, relu), MeanPool((1, 1)), GlobalMeanPool(),
                    flatten, Dense(4 => 2))
        for (model, input) in ((mlp, x), (cnn, xc))
            model(input)
            opt_state = setup(Adam(), model)
            grads = gradient(m -> logitcrossentropy(m(input), y), model)
            update!(opt_state, model, grads[1])
            withgradient(m -> mse(m(input), yf), model)
            train!((m, x, y) -> logitcrossentropy(m(x), y), model, [(input, y)], opt_state)
        end
        # The GC pacers' registries hold `WeakRef`s and `objectid` hashes, which would mean nothing
        # once the package image is loaded.
        empty!(Train.STEP_PACERS)
        empty!(Train.STEP_TYPES)
    end
end
