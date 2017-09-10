```julia
Flux.train!(loss, repeated((x,y), 1000), SGD(params(m), 0.1),
            cb = throttle(() -> @show(loss(x, y)), 5))
```
