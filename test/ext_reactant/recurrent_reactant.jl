struct TupleStateTestCell{I, H}
    Wi::I
    Wh::H
end

Flux.@layer TupleStateTestCell

Flux.initialstates(cell::TupleStateTestCell) =
    (zero.(cell.Wh[:, 1]), zero.(cell.Wh[:, 1]))

function (cell::TupleStateTestCell)(x::AbstractVecOrMat, (h, c))
    h_new = tanh.(cell.Wi * x .+ cell.Wh * h)
    c_new = tanh.(c .+ h_new)
    return h_new, (h_new, c_new)
end

_recurrent_state_loss(state::AbstractArray) = sum(abs2, state)
_recurrent_state_loss(state::Tuple) = sum(_recurrent_state_loss, state)

_recurrent_state_isapprox(a::AbstractArray, b::AbstractArray) = isapprox(a, b; rtol = 1e-4, atol = 1e-5)
_recurrent_state_isapprox(a::Tuple, b::Tuple) =
    length(a) == length(b) && all(_recurrent_state_isapprox(x, y) for (x, y) in zip(a, b))

function recurrent_return_state_loss(model, x, state)
    y, final_state = model(x, state)
    return sum(abs2, y) + _recurrent_state_loss(final_state)
end

_recurrent_host_state(state::AbstractArray) = Array(state)
_recurrent_host_state(state::Tuple) = map(_recurrent_host_state, state)

_apply_recurrent(model, x, state) = model(x, state)
_apply_model(model, x) = model(x)

@testset "Reactant traced recurrent scan" begin
    @testset "HLO contains a loop, without growing concatenations" begin
        cell = RNNCell(4 => 8)
        x = Reactant.to_rarray(randn(Float32, 4, 50))
        state = Reactant.to_rarray(randn(Float32, 8))
        hlo = sprint(show, Reactant.@code_hlo Flux.scan(cell, x, state))
        @test occursin("stablehlo.while", hlo)
        @test !occursin("stablehlo.concatenate", hlo)
    end

    @testset "lazy Dense reshape can feed a recurrent layer" begin
        model = Chain(Dense(3 => 4), RNN(4 => 4), LSTM(4 => 5), GRU(5 => 4))
        x = randn(Float32, 3, 6, 2)
        expected = model(x)
        model_re, x_re = (model, x) |> MLDataDevices.reactant_device(force = true)
        compiled = Reactant.@compile _apply_model(model_re, x_re)
        @test Array(compiled(model_re, x_re)) ≈ expected rtol=1e-4 atol=1e-5
    end

    @testset "single-step and empty sequences" begin
        cell = RNNCell(2 => 3)
        state = Reactant.to_rarray(zeros(Float32, 3))
        x = Reactant.to_rarray(ones(Float32, 2, 1))
        compiled = Reactant.@compile Flux.scan(cell, x, state)
        output, _ = compiled(cell, x, state)
        @test size(output) == (3, 1)

        empty_x = Reactant.to_rarray(zeros(Float32, 2, 0))
        @test_throws BoundsError Reactant.@compile Flux.scan(cell, empty_x, state)
    end

    @testset "outputs, final states and initial-state preservation" begin
        x = randn(Float32, 4, 12, 2)
        rnn = RNN(RNNCell(4 => 3); return_state = true)
        lstm = LSTM(LSTMCell(4 => 3); return_state = true)
        gru = GRU(GRUCell(4 => 3); return_state = true)
        custom = Recurrence(TupleStateTestCell(randn(Float32, 3, 4), randn(Float32, 3, 3)); return_state = true)
        cases = (
            (rnn, RNN(rnn.cell), (randn(Float32, 3, 2),), "RNN"),
            (lstm, LSTM(lstm.cell), ((randn(Float32, 3, 2), randn(Float32, 3, 2)),), "LSTM"),
            (gru, GRU(gru.cell), (randn(Float32, 3, 2),), "GRU"),
            (custom, Recurrence(custom.cell), ((randn(Float32, 3, 2), randn(Float32, 3, 2)),), "custom tuple state"),
        )
        dev = MLDataDevices.reactant_device(force = true)

        for (with_state, without_state, state_arg, name) in cases
            state = only(state_arg)
            expected_y, expected_state = with_state(x, state)
            model_re, x_re, state_re = (with_state, x, state) |> dev
            initial_state_before = _recurrent_host_state(state_re)
            compiled = Reactant.@compile recurrent_return_state_loss(model_re, x_re, state_re)
            # Compile a separate forward executable so the returned arrays and tuple states can be
            # checked directly against eager Flux, including both return_state settings.
            forward = Reactant.@compile _apply_recurrent(model_re, x_re, state_re)
            actual_y, actual_state = forward(model_re, x_re, state_re)
            @test Array(actual_y) ≈ expected_y rtol=1e-4 atol=1e-5
            @test _recurrent_state_isapprox(_recurrent_host_state(actual_state), expected_state)
            @test _recurrent_host_state(state_re) == initial_state_before

            model_without_state_re = without_state |> dev
            forward_without_state = Reactant.@compile _apply_recurrent(model_without_state_re, x_re, state_re)
            actual_y_only = forward_without_state(model_without_state_re, x_re, state_re)
            @test Array(actual_y_only) ≈ expected_y rtol=1e-4 atol=1e-5
            @test compiled(model_re, x_re, state_re) isa Reactant.ConcreteRNumber
        end
    end

    @testset "parameter, input and caller-state gradients" begin
        x = randn(Float32, 4, 5, 2)
        cells = (
            (RNN(RNNCell(4 => 3); return_state = true), (randn(Float32, 3, 2),), "RNN"),
            (LSTM(LSTMCell(4 => 3); return_state = true), ((randn(Float32, 3, 2), randn(Float32, 3, 2)),), "LSTM"),
            (GRU(GRUCell(4 => 3); return_state = true), (randn(Float32, 3, 2),), "GRU"),
            (Recurrence(TupleStateTestCell(randn(Float32, 3, 4), randn(Float32, 3, 3)); return_state = true),
             ((randn(Float32, 3, 2), randn(Float32, 3, 2)),), "custom tuple state"),
        )
        dev = MLDataDevices.reactant_device(force = true)
        for (model, state_arg, name) in cells
            @testset "$name gradients" begin
                state = only(state_arg)
                ref_loss, ref_grad = Flux.withgradient(
                    recurrent_return_state_loss, AutoZygote(), model, x, state,
                )
                model_re, x_re, state_re = (model, x, state) |> dev
                reactant_loss_value, reactant_grad = reactant_withgradient(
                    recurrent_return_state_loss, model_re, x_re, state_re,
                )
                @test reactant_loss_value ≈ ref_loss rtol=1e-4 atol=1e-5
                check_equal_leaves(ref_grad, reactant_grad |> cpu_device(); rtol=1e-4, atol=1e-5)
            end
        end
    end
end
