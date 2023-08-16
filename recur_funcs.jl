using Flux

function run_new_recur()
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


  x_block = reshape(reduce(vcat, x), 1, 1, length(x))
  nm_layer = Flux.NewRecur(cell; return_sequence = true)
  _out = layer(x_block)
  e, g = Flux.withgradient(nm_layer) do layer
    out = layer(x_block)
    sum(out[1, 1, end])
  end
  grads = g[1][:cell]

  @show primal[1] ≈ e
  @show ∇Wi ≈ grads[:Wi]
  @show ∇Wh ≈ grads[:Wh]
  @show ∇b ≈ grads[:b]
  @show ∇state0 ≈ grads[:state0]

  return
end

function run_scan_full()

  x = [[2.0f0], [3.0f0], [4.0f0]]
  x_block = reshape(reduce(vcat, x), 1, 1, length(x))
  # nm_layer = Flux.NewRecur(cell; return_sequence = true)
  w = zeros(1)
  _out = Flux.scan_full((a, b)->(sum(w.*b), sum(w.*b)), 0.0f0, x_block)
  e, g = Flux.withgradient(w) do layer
    out = Flux.scan_full((a, b)->(sum(w.*b), sum(w.*b)), 0.0f0, x_block)
    sum(out[1, 1, end])
  end
  grads = g[1][:cell]
  return
end
