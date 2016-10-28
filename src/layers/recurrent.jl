@net type Recurrent
  Wxh; Whh; Why
  bh; by
  hidden
  function (x)
    hidden = σ( x * Wxh + hidden * Whh + bh )
    y = hidden * Why + by
  end
end

Recurrent(in::Integer, hidden::Integer, out::Integer; init = initn) =
  Recurrent(init((in, hidden)), init((hidden, hidden)), init((hidden, out)),
            init(hidden), init(out), zeros(Float32, hidden))
