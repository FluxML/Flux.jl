export Recurrent

@net type Recurrent
  Wxy; Wyy; by
  y
  function (x)
    y = tanh( x * Wxy + y * Wyy + by )
  end
end

Recurrent(in, out; init = initn) =
  Recurrent(init((in, out)), init((out, out)), init(out), init(out))
