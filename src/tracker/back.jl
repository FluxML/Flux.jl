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
  foreach(back, c.args, data.(Δs))
end

back_(::Call{Void}, Δ) = nothing

accum!(x, Δ) = x .+ Δ
accum!(x::AbstractArray, Δ) = (x .+= Δ)

function back(x::Tracked, Δ)
  x.isleaf && (x.grad = accum!(x.grad, Δ); return)
  ref = x.ref -= 1
  if ref > 0 || isdefined(x, :grad)
    if isdefined(x, :grad)
      x.grad = accum!(x.grad, Δ)
    else
      x.grad = Δ
    end
    ref == 0 && back_(x.f, x.grad)
  else
    ref == 0 && back_(x.f, Δ)
  end
  return
end

back(::Void, _) = return

# Interface methods

# TODO: if an error occurs in `back` the refcounts will be broken
# and `back` will silently fail to update.
# Refcounts are also probably not safe in some situations (e.g. back called
# from within a backpropagator)

function back!(x, Δ)
  istracked(x) || return
  scan(x)
  back(tracker(x), Δ)
  return
end

# Out-of-place gradients

struct Params
  params::IdSet
  Params(xs) = new(IdSet(xs))
end

@forward Params.params Base.start, Base.next, Base.done

function Base.show(io::IO, ps::Params)
  print(io, "Params([")
  join(io, ps.params, ", ")
  print(io, "])")
end

struct Grads
  grads::ObjectIdDict
end

Base.show(io::IO, ps::Grads) = println(io, "Grads(...)")

Grads() = Grads(ObjectIdDict())

Grads(ps::Params) = Grads(ObjectIdDict(tracker(p) => init_grad(data(p)) for p in ps))

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
  foreach((x, Δ) -> back(g, x, Δ), c.args, Δs)
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

back(::Grads, ::Void, _) = return

function forward(f, ps::Params)
  y = f()
  y, function (Δ)
    g = Grads(ps)
    if istracked(y)
      scan(y)
      back(g, tracker(y), Δ)
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
