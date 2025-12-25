```@meta
CollapsedDocStrings = true
```
# [Callback Helpers](@id man-callback-helpers)

```@docs
Flux.throttle
```

## Patience Helpers

Flux provides utilities for controlling your training procedure according to some monitored condition and a maximum `patience`. For example, you can use `early_stopping` to stop training when the model is converging or deteriorating, or you can use `plateau` to check if the model is stagnating.

For example, below we create a pseudo-loss function that decreases, bottoms out, and then increases. The early stopping trigger will break the loop before the loss increases too much.
```julia
# create a pseudo-loss that decreases for 4 calls, then starts increasing
# we call this like loss()
loss = let t = 0
  () -> begin
    t += 1
    (t - 4) ^ 2
  end
end

# create an early stopping trigger
# returns true when the loss increases for two consecutive steps
es = early_stopping(loss, 2; init_score = 9)

# this will stop at the 6th (4 decreasing + 2 increasing calls) epoch
for epoch in 1:10
  es() && break
end
```

The keyword argument `distance` of `early_stopping` is a function of the form `distance(best_score, score)`. By default `distance` is `-` and `init_score` is `Inf`, meaning that the monitored metric `f` is expected to be decreasing and minimized. If you use a metric such that improvement is shown by increasing values (e.g. accuracy), you can customize the `distance` function and the `init_score` value to, for example, `(best_score, score) -> score - best_score` and `-Inf`, respectively.
```julia
# create a pseudo-accuracy that increases by 0.01 each time from 0 to 1
# we call this like acc()
acc = let v = 0
  () -> v = max(1, v + 0.01)
end

# create an early stopping trigger for accuracy
es = early_stopping(acc, 3; delta = (best_score, score) -> score - best_score, init_score = -Inf)

# this will iterate until the 10th epoch
for epoch in 1:10
  es() && break
end
```

`early_stopping` and `plateau` are both built on top of `patience`. You can use `patience` to build your own triggers that use a patient counter. For example, if you want to trigger when the loss is below a threshold for several consecutive iterations:
```julia
threshold(f, thresh, delay) = patience(delay) do
  f() < thresh
end
```

Both `predicate` in `patience` and `f` in `early_stopping` / `plateau` can accept extra arguments. You can pass such extra arguments to `predicate` or `f` through the returned function:
```julia
trigger = patience((a; b) -> a > b, 3)

# this will iterate until the 10th epoch
for epoch in 1:10
  trigger(1; b = 2) && break
end

# this will stop at the 3rd epoch
for epoch in 1:10
  trigger(3; b = 2) && break
end
```

```@docs
Flux.patience
Flux.early_stopping
Flux.plateau
```
