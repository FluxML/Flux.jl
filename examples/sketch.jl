# Simple Perceptron Layer

@flux type Simple
  weight
  bias
  x -> σ( weight*x + bias )
end

Simple(nx::Integer, ny::Integer; init = randn) =
  Simple(init(nx, ny), init(ny))

# Time Delay Node

type Delay
  n::Int
  next
end

# feed(l::Delay, x) = ...

# back(l::Delay, y) = ...

# Simple Recurrent

@flux type RecurrentU
  Wxh; Whh; Bh
  Wxy; Why; By

  function feed(x, hidden)
    hidden′ = σ( Wxh*x + Whh*hidden + Bh )
    y = σ( Wxy*x + Why*hidden′ + By )
    y, hidden′
  end
end

Recurrent(nx, ny, nh; init = randn) =
  Recurrent(init(nx, nh), init(nh, nh), init(nh),
            init(nx, ny), init(nh, ny), init(ny))

@flux type Looped{T}
  delay::Delay
  layer::T

  function (x)
    y, hidden = layer(x, delay(hidden))
    return y
  end
end

type Recurrent
  layer::Looped{RecurrentU}
end

Recurrent(nx, ny, nh; init = randn, delay = 10) =
  Looped(Delay(delay, init(nh)), RecurrentU(nx, ny, nh))

@forward Recurrent.layer feed
