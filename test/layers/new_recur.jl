@testset "NewRecur RNN" begin
  @testset "Forward Pass" begin
    # tanh is needed for forward check to determine ordering of inputs.
    cell = Flux.RNNCell(1, 1, tanh)
    layer = Flux.NewRecur(cell; return_sequence=true)
    layer.cell.Wi .= 5.0
    layer.cell.Wh .= 4.0
    layer.cell.b .= 0.0f0
    layer.cell.state0 .= 7.0
    x = reshape([2.0f0, 3.0f0], 1, 1, 2)

    # Lets make sure th output is correct
    h = cell.state0
    h, out = cell(h, [2.0f0])
    h, out = cell(h, [3.0f0])
    
    @test eltype(layer(x)) <: Float32
    @test size(layer(x)) == (1, 1, 2)
    @test layer(x)[1, 1, 2] ≈ out[1,1]

    @test length(layer(cell.state0, x)) == 2 # should return a tuple. Maybe better test is needed.
    @test layer(cell.state0, x)[2][1,1,2] ≈ out[1,1]

    @test_throws MethodError layer([2.0f0])
    @test_throws MethodError layer([2.0f0;; 3.0f0])
  end

  @testset "gradients-implicit" begin
    cell = Flux.RNNCell(1, 1, identity)
    layer = Flux.Recur(cell)
    layer.cell.Wi .= 5.0
    layer.cell.Wh .= 4.0
    layer.cell.b .= 0.0f0
    layer.cell.state0 .= 7.0
    x = [[2.0f0], [3.0f0]]

    # theoretical primal gradients
    primal =
      layer.cell.Wh .* (layer.cell.Wh * layer.cell.state0 .+ x[1] .* layer.cell.Wi) .+
      x[2] .* layer.cell.Wi
    ∇Wi = x[1] .* layer.cell.Wh .+ x[2]
    ∇Wh = 2 .* layer.cell.Wh .* layer.cell.state0 .+ x[1] .* layer.cell.Wi
    ∇b = layer.cell.Wh .+ 1
    ∇state0 = layer.cell.Wh .^ 2

    nm_layer = Flux.NewRecur(cell; return_sequence = true)
    ps = Flux.params(nm_layer)
    x_block = reshape(vcat(x...), 1, 1, length(x))
    e, g = Flux.withgradient(ps) do
      out = nm_layer(x_block)
      sum(out[1, 1, 2])
    end
    
    @test primal[1] ≈ e
    @test ∇Wi ≈ g[ps[1]]
    @test ∇Wh ≈ g[ps[2]]
    @test ∇b ≈ g[ps[3]]
    @test ∇state0 ≈ g[ps[4]]
  end

  @testset "gradients-explicit" begin

    cell = Flux.RNNCell(1, 1, identity)
    layer = Flux.Recur(cell)
    layer.cell.Wi .= 5.0
    layer.cell.Wh .= 4.0
    layer.cell.b .= 0.0f0
    layer.cell.state0 .= 7.0
    x = [[2.0f0], [3.0f0]]

    # theoretical primal gradients
    primal =
      layer.cell.Wh .* (layer.cell.Wh * layer.cell.state0 .+ x[1] .* layer.cell.Wi) .+
      x[2] .* layer.cell.Wi
    ∇Wi = x[1] .* layer.cell.Wh .+ x[2]
    ∇Wh = 2 .* layer.cell.Wh .* layer.cell.state0 .+ x[1] .* layer.cell.Wi
    ∇b = layer.cell.Wh .+ 1
    ∇state0 = layer.cell.Wh .^ 2


    x_block = reshape(vcat(x...), 1, 1, length(x))
    nm_layer = Flux.NewRecur(cell; return_sequence = true)
    e, g = Flux.withgradient(nm_layer) do layer
      out = layer(x_block)
      sum(out[1, 1, 2])
    end
    grads = g[1][:cell]

    @test primal[1] ≈ e
    @test ∇Wi ≈ grads[:Wi]
    @test ∇Wh ≈ grads[:Wh]
    @test ∇b ≈ grads[:b]
    @test ∇state0 ≈ grads[:state0]
  end
end

@testset "New Recur RNN Partial Sequence" begin
  @testset "Forward Pass" begin
    cell = Flux.RNNCell(1, 1, identity)
    layer = Flux.NewRecur(cell)
    layer.cell.Wi .= 5.0
    layer.cell.Wh .= 4.0
    layer.cell.b .= 0.0f0
    layer.cell.state0 .= 7.0
    x = reshape([2.0f0, 3.0f0], 1, 1, 2)

    h = cell.state0
    h, out = cell(h, [2.0f0])
    h, out = cell(h, [3.0f0])
    
    @test eltype(layer(x)) <: Float32
    @test size(layer(x)) == (1, 1)
    @test layer(x)[1, 1] ≈ out[1,1]
    
    @test length(layer(cell.state0, x)) == 2
    @test layer(cell.state0, x)[2][1,1] ≈ out[1,1]
    
    @test_throws MethodError layer([2.0f0])
    @test_throws MethodError layer([2.0f0;; 3.0f0])
  end

  @testset "gradients-implicit" begin
    cell = Flux.RNNCell(1, 1, identity)
    layer = Flux.Recur(cell)
    layer.cell.Wi .= 5.0
    layer.cell.Wh .= 4.0
    layer.cell.b .= 0.0f0
    layer.cell.state0 .= 7.0
    x = [[2.0f0], [3.0f0]]

    # theoretical primal gradients
    primal =
      layer.cell.Wh .* (layer.cell.Wh * layer.cell.state0 .+ x[1] .* layer.cell.Wi) .+
      x[2] .* layer.cell.Wi
    ∇Wi = x[1] .* layer.cell.Wh .+ x[2]
    ∇Wh = 2 .* layer.cell.Wh .* layer.cell.state0 .+ x[1] .* layer.cell.Wi
    ∇b = layer.cell.Wh .+ 1
    ∇state0 = layer.cell.Wh .^ 2

    nm_layer = Flux.NewRecur(cell; return_sequence = false)
    ps = Flux.params(nm_layer)
    x_block = reshape(vcat(x...), 1, 1, length(x))
    e, g = Flux.withgradient(ps) do
      out = (nm_layer)(x_block)
      sum(out)
    end
    
    @test primal[1] ≈ e
    @test ∇Wi ≈ g[ps[1]]
    @test ∇Wh ≈ g[ps[2]]
    @test ∇b ≈ g[ps[3]]
    @test ∇state0 ≈ g[ps[4]]
  end

  @testset "gradients-explicit" begin
    cell = Flux.RNNCell(1, 1, identity)
    layer = Flux.Recur(cell)
    layer.cell.Wi .= 5.0
    layer.cell.Wh .= 4.0
    layer.cell.b .= 0.0f0
    layer.cell.state0 .= 7.0
    x = [[2.0f0], [3.0f0]]

    # theoretical primal gradients
    primal =
      layer.cell.Wh .* (layer.cell.Wh * layer.cell.state0 .+ x[1] .* layer.cell.Wi) .+
      x[2] .* layer.cell.Wi
    ∇Wi = x[1] .* layer.cell.Wh .+ x[2]
    ∇Wh = 2 .* layer.cell.Wh .* layer.cell.state0 .+ x[1] .* layer.cell.Wi
    ∇b = layer.cell.Wh .+ 1
    ∇state0 = layer.cell.Wh .^ 2

    x_block = reshape(vcat(x...), 1, 1, length(x))
    nm_layer = Flux.NewRecur(cell; return_sequence = false)
    e, g = Flux.withgradient(nm_layer) do layer
      out = layer(x_block)
      sum(out)
    end
    grads = g[1][:cell]

    @test primal[1] ≈ e
    @test ∇Wi ≈ grads[:Wi]
    @test ∇Wh ≈ grads[:Wh]
    @test ∇b ≈ grads[:b]
    @test ∇state0 ≈ grads[:state0]

  end
end
