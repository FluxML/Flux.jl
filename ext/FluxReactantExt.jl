module FluxReactantExt

using Flux: Flux, Train
using Optimisers: Optimisers
using ADTypes: AutoEnzyme
using ProgressLogging: @withprogress, @logprogress
import Reactant

# Fixed-arity compiled step: the whole batch is passed as ONE tuple (splat inside), so the
# thunk arity is always 4 and stable regardless of how many arrays a batch holds. Mirrors
# Flux's own `_train_step!` (src/train.jl). Returns the primal loss as an on-device scalar
# (computing it is free — `withgradient` already runs a reverse-with-primal pass); `_reactant_train!`
# reads it each step for the `isfinite` guard and the progress bar.
function _reactant_step!(loss, opt, model, data::Tuple)
    l, gs = Flux.withgradient(m -> loss(m, data...), AutoEnzyme(), model)
    Optimisers.update!(opt, model, gs[1])   # folded into the executable; mutates in place
    return l
end

# Compiled steps are cached across `train!` calls so repeated calls — one per epoch, or a
# benchmark's separate warm-up and timed runs — reuse the XLA executable instead of recompiling
# (Reactant has no global compile cache; `@compile` recompiles on every call). The model and
# optimiser state are runtime inputs to the executable, so a step compiled once is valid for
# every later call that passes the same objects.
#
# The key identifies exactly what a compiled executable is specialised to:
#   * `objectid(model)` + `typeof(model)` — the model's parameter array shapes are baked into the
#     executable but are *not* part of the model's type (e.g. `Dense(2=>3)` and `Dense(2=>5)` share
#     a type), so we pin the concrete model object; its parameters keep their identity across the
#     in-place updates, so the id is stable for the model's lifetime.
#   * `typeof(opt)` — the optimiser *rule* structure. Its scalars (η, βt, …) are on-device runtime
#     inputs, so one executable serves any learning rate.
#   * `objectid(loss)` — keyed by *identity*, not type, on purpose: two closures of the same type
#     may capture different constants that Reactant bakes into the executable.
#   * batch element types + sizes — a differently-shaped batch needs its own executable.
#
# The cache holds no strong reference to the model/opt/loss (only their `objectid`s), so it never
# keeps a model (and its parameters) alive; only the small compiled-executable handles are retained
# for the session. Identity keying is conservative — a fresh model object recompiles once — but
# never reuses a step compiled for a different loss, optimiser, or batch shape.
const COMPILE_CACHE = Dict{Any, Any}()

function _compiled_step(loss, opt, model, dtuple)
    key = (objectid(model), typeof(model), typeof(opt), objectid(loss),
           map(typeof, dtuple), map(size, dtuple))
    return get!(COMPILE_CACHE, key) do
        Reactant.@compile _reactant_step!(loss, opt, model, dtuple)
    end
end

# Reactant/XLA implementation of `Flux.train!`, dispatched to from the ReactantDevice branch of the
# eager `train!` (src/train.jl). The model and optimiser state already live on the Reactant
# device, and the data is required to be device-resident too. The fused step is compiled once per
# distinct (model, optimiser, loss, batch-shape) and reused, mutating `opt`/`model` in place.
function Train._reactant_train!(loss, model, data, opt)
    Flux.trainmode!(model)
    @withprogress for (i, d) in enumerate(data)
        dtuple = d isa Tuple ? d : (d,)
        Flux.get_device_type(dtuple) <: Flux.ReactantDevice || throw(ArgumentError(
            "`train!` on a Reactant model requires device-resident data; move each batch to the \
             model's device first, e.g. `data = [(x, y) |> reactant_device() for (x, y) in data]`."))
        step = _compiled_step(loss, opt, model, dtuple)
        l = step(loss, opt, model, dtuple)   # opt/model updated in place inside the executable
        lh = Reactant.to_number(l)           # host read forces the async device→host sync
        isfinite(lh) || throw(DomainError(lazy"Loss is $lh on data item $i, stopping training"))
        @logprogress Base.haslength(data) ? i / length(data) : nothing
    end
    return nothing
end

end # module
