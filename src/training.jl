using Juno: info
using .Batches: tobatch

"""
  @cb for ... end t expr

Run the for loop, executing `expr` every `t` seconds.
"""
macro cb(ex, t, f)
  @assert isexpr(ex, :for)
  cond, body = ex.args
  @esc t f cond body
  :(let
    t0 = time_ns()
    dt = $t*1e9
    @progress $(Expr(:for, cond, quote
      t = time_ns()
      if t - t0 > dt
        t0 = t
        f = () -> $f
        f()
      end
      $body
    end))
  end)
end

function train!(m, train; cb = [],
                epoch = 1, η = 0.1, loss = mse)
    @progress for e in 1:epoch
      info("Epoch $e")
      @cb for (x, y) in train
        x, y = mapt(tobatch, (x, y))
        ŷ = m(x)
        any(isnan, ŷ) && error("NaN")
        Δ = back!(loss, 1, ŷ, y)
        back!(m, Δ, x)
        update!(m, η)
      end 5 foreach(f -> f(), cb)
    end
    return m
end
