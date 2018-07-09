init_grad(x) = zero(x)
zero_grad!(x) = zero(x)
zero_grad!(x::AbstractArray) = (x .= 0)

scan(c::Call) = foreach(scan, c.args)

function scan(x::Tracked)
  x.isleaf && return
  ref = x.ref += 1
  if ref == 1
    scan(x.f)
    isdefined(x, :grad) && (x.grad = zero_grad!(x.grad))
  else
    isdefined(x, :grad) || (x.grad = init_grad(x.data))
  end
  return
end

function scan(x)
  istracked(x) && scan(tracker(x))
  return
end

function back_(c::Call, Δ)
  Δs = c.func(Δ)
  (Δs isa Tuple && length(Δs) >= length(c.args)) ||
    error("Gradient is not a tuple of length $(length(c.args))")
  foreach((x, Δ) -> istracked(x) && back(x, Δ), c.args, Δs)
end

back_(::Call{Void}, Δ) = nothing

accum!(x, Δ) = x .+ Δ
accum!(x::AbstractArray, Δ) = (x .+= Δ)

function back(x::Tracked, Δ)
  x.isleaf && (x.grad = accum!(x.grad, Δ); return)
  ref = x.ref -= 1
  if isdefined(x, :grad)
    x.grad = accum!(x.grad, Δ)
    ref == 0 && back_(x.f, x.grad)
  else
    ref == 0 && back_(x.f, Δ)
  end
  return
end

back(x, Δ) = back(tracker(x), Δ)
back(x::Void, Δ) = error("Can't backpropagate through `nothing`")

# Interface methods

# TODO: if an error occurs in `back` the refcounts will be broken
# and `back` will silently fail to update.
# Refcounts are also probably not safe in some situations (e.g. back called
# from within a backpropagator)

function back!(x::Tracked, Δ)
  scan(x)
  back(x, Δ)
end

back!(x, Δ) = back!(tracker(x), Δ)

# Out-of-place gradients

struct Params
  params::IdSet
  Params(xs) = new(IdSet(xs))
end

@forward Params.params Base.start, Base.next, Base.done

struct Grads
  grads::ObjectIdDict
end

Grads() = Grads(ObjectIdDict())

Base.getindex(g::Grads, x::Tracked) = g.grads[x]
function Base.getindex(g::Grads, x)
  istracked(x) || error("Object not tracked: $x")
  g[tracker(x)]
end

@forward Grads.grads Base.setindex!, Base.haskey

accum!(g::Grads, x, Δ) = g[x] = haskey(g, x) ? g[x] + Δ : Δ

function back_(g::Grads, c::Call, Δ)
  Δs = c.func(Δ)
  (Δs isa Tuple && length(Δs) >= length(c.args)) ||
    error("Gradient is not a tuple of length $(length(c.args))")
  foreach((x, Δ) -> istracked(x) && back(g, x, Δ), c.args, Δs)
end

back_(g::Grads, ::Call{Void}, Δ) = nothing

function back(g::Grads, x::Tracked, Δ)
  x.isleaf && (accum!(g, x, Δ); return)
  ref = x.ref -= 1
  if ref > 0 || haskey(g, x)
    accum!(g, x, Δ)
    ref == 0 && back_(g, x.f, g[x])
  else
    ref == 0 && back_(g, x.f, Δ)
  end
  return
end

back(g::Grads, x, Δ) = back(g, tracker(x), Δ)
back(g::Grads, x::Void, Δ) = error("Can't backpropagate through `nothing`")

function forward(f, ps::Params)
  y = f()
  y, function (Δ)
    g = Grads()
    if istracked(y)
      scan(y)
      back(g, y, Δ)
    end
    for p in ps
      haskey(g, tracker(p)) ||
        (g[tracker(p)] = init_grad(data(p)))
    end
    return g
  end
end

function forward(f, args...)
  args = param.(args)
  y, back = forward(() -> f(args...), Params(args))
  y, Δ -> getindex.(back(Δ), args)
end

function losscheck(x)
  x isa Real || error("Function output is not scalar")
  isinf(x) && error("Loss is infinite")
  isnan(x) && error("Loss is NaN")
end

function gradient(f, args...)
  y, back = forward(f, args...)
  losscheck(y)
  return back(1)
end

derivative(f, x) = gradient(f, x)[1]
