using Flux: with_rotary_position_embedding

@testset "Rotary Position Embedding Tests" begin
    Random.seed!(123)
    test_sizes = [(2,2), (4,6), (8,10)]

    for (n, d) in test_sizes
        x = randn(n, d)
        test_gradients(
            with_rotary_position_embedding,
            x;
            rtol=1e-4,
            atol=1e-4,
            test_gpu=false,
            compare_finite_diff=true,
            loss=(f, x) -> sum(f(x))
        )
    end

    # Edge cases
    test_gradients(
        with_rotary_position_embedding,
        zeros(4, 6);
        loss=(f, x) -> sum(f(x))
    )

    test_gradients(
        with_rotary_position_embedding,
        ones(4, 6);
        loss=(f, x) -> sum(f(x))
    )
end
