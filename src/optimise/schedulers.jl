"""
    Scheduler{T<:ParameterSchedulers.AbstractSchedule, O, F}
    Scheduler(schedule::ParameterSchedulers.AbstractSchedule, opt, update_func)
    Scheduler(schedule::ParameterSchedulers.AbstractSchedule, opt;
                   update_func = (o, s) -> (o.eta = s))

Wrap a `schedule` and `opt` together into a `Scheduler`.
The `schedule` is iterated each time the optimizer updates the gradients.
The `Scheduler` can be used anywhere a Flux optimizer is used.

The keyword argument constructor sets `update_func(opt, s)`
to schedule the learning rate of `opt` to `s` on every iteration.
You can update any field of `opt` by passing your own `update_func`.
*Note: [`ADADelta`](@ref) does not have a learning rate,
       so there is no default `update_func`*

# Arguments
- `schedule::ParameterSchedulers.AbstractSchedule`: the schedule to use
- `opt`: a Flux optimizer
- `update_func`: a mutating function of with inputs `(optim, param)`
                 that updates `optim` based on the current `param` value
"""
mutable struct Scheduler{T<:AbstractSchedule, O, F}
  schedule::T
  optim::O
  update_func::F
end

for Opt in (Descent, ADAM, Momentum, Nesterov, RMSProp,
            ADAGrad, AdaMax, AMSGrad, NADAM,
            RADAM, OADAM, AdaBelief)
  @eval begin
  Scheduler(schedule::AbstractSchedule, opt::$Opt; update_func = (o, s) -> (o.eta = s)) =
    Scheduler(schedule, opt, update_func)
  end
end

Optimise.Optimisers.init(o::Scheduler, x::AbstractArray) =
  (t = 1, optim = init(o.optim, x))

function Optimise.Optimisers.apply!(opt::Scheduler, x, dx, state)
  # set param
  opt.update_func(opt.optim, opt.schedule[state.t])

  # do normal apply
  dx, s = Optimise.Optimisers.apply!(opt.optim, x, dx, state.optim)

  return dx, (t = t + 1, optim = s)
end