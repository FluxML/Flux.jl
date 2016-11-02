export Recurrent, GatedRecurrent, LSTM

@net type Recurrent
  Wxy; Wyy; by
  y
  function (x)
    y = tanh( x * Wxy + y * Wyy + by )
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
    reset  = σ( x * Wxr + y * Wyr + br )
    update = σ( x * Wxu + y * Wyu + bu )
    y′ = tanh( x * Wxh + (reset .* y) * Wyh + bh )
    y = (1 .- update) .* y′ + update .* y
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
    forget = σ( x * Wxf + y * Wyf + bf )
    input  = σ( x * Wxi + y * Wyi + bi )
    output = σ( x * Wxo + y * Wyo + bo )
    # State update and output
    state′ = tanh( x * Wxc + y * Wyc + bc )
    state  = forget .* state + input .* state′
    y = output .* tanh(state)
  end
end

LSTM(in, out; init = initn) =
  LSTM(vcat([[init((in, out)), init((out, out)), init(out)] for _ = 1:4]...)...,
       zeros(Float32, out), zeros(Float32, out))
