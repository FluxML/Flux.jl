export Recurrent, GatedRecurrent, LSTM

@net type Recurrent
  Wxy; Wyy; by
  y
  function (x)
    y = tanh( x * Wxy + y{-1} * Wyy + by )
  end
end

Recurrent(in, out; init = initn) =
  Recurrent(init((in, out)), init((out, out)), init(out), init(out))

@net type GatedRecurrent
  Wxr; Wyr; br
  Wxu; Wyu; bu
  Wxh; Wyh; bh
  y
  function (x)
    reset  = σ( x * Wxr + y{-1} * Wyr + br )
    update = σ( x * Wxu + y{-1} * Wyu + bu )
    y′ = tanh( x * Wxh + (reset .* y{-1}) * Wyh + bh )
    y = (1 .- update) .* y′ + update .* y{-1}
  end
end

GatedRecurrent(in, out; init = initn) =
  GatedRecurrent(vcat([[init((in, out)), init((out, out)), init(out)] for _ = 1:3]...)...,
       zeros(Float32, out))

@net type LSTM
  Wxf; Wyf; bf
  Wxi; Wyi; bi
  Wxo; Wyo; bo
  Wxc; Wyc; bc
  y; state
  function (x)
    # Gates
    forget = σ( x * Wxf + y{-1} * Wyf + bf )
    input  = σ( x * Wxi + y{-1} * Wyi + bi )
    output = σ( x * Wxo + y{-1} * Wyo + bo )
    # State update and output
    state′ = tanh( x * Wxc + y{-1} * Wyc + bc )
    state  = forget .* state{-1} + input .* state′
    y = output .* tanh(state)
  end
end

LSTM(in, out; init = initn) =
  LSTM(vcat([[init((in, out)), init((out, out)), init((1, out))] for _ = 1:4]...)...,
       zeros(Float32, out), zeros(Float32, out))
