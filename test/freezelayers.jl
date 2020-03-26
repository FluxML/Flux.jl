using Flux: Chain, Dense, softmax, params, train!
using Flux: crossentropy, ADAM, freezelayers!
using Test

function collect_params(m, layer_indexes::Vector{Int64})
    ps = []
    for layer_index in layer_indexes
        layer = m.layers[layer_index]
        param_names = fieldnames(typeof(layer))
        for param_name in param_names
            p = deepcopy(getfield(layer, param_name))
            push!(ps, p)
        end
    end

    return ps
end

@testset "FreezeLayers" begin
    # Fixed layers
    freezed_layer_indexes = [1, 3]
    mutable_layer_indexes = [2, 4]

    # Model
    m = Chain(
        Dense(4, 2),
        Dense(2, 3),
        Dense(3, 4),
        Dense(4, 3),
        softmax
    )

    # Original params
    ps_freezed = collect_params(m, freezed_layer_indexes)
    ps_mutable = collect_params(m, mutable_layer_indexes)

    # Update params
    train!(
        (x, y) -> crossentropy(m(x), y),
        freezelayers!(params(m), m, freezed_layer_indexes),
        [(rand(4, 10), rand(1, 10))],
        ADAM(0.005)
    )

    # Params after update
    new_ps_freezed = collect_params(m, freezed_layer_indexes)
    new_ps_mutable = collect_params(m, mutable_layer_indexes)

    @test new_ps_freezed == ps_freezed
    @test new_ps_mutable != ps_mutable
end
