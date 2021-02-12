"""
    Scheduler{T<:ParameterSchedulers.AbstractSchedule, O}
    Scheduler(schedule::ParameterSchedulers.AbstractSchedule, opt)

Wrap a `schedule` and `opt` together with a `Scheduler`.
Each time `Optimise.update!` is called, the scheduler sets the
learning rate (`opt.eta`) to the current schedule value,
then the schedule is advanced by one iteration.

A `Scheduler` can be used anywhere a Flux optimizer is used.

!!! warning
  `opt` is limited to optimizer with a learning rate
  (i.e. the field `opt.eta` exists)

# Examples
```jldoctest
julia> opt = Momentum();

julia> schedule = Schedule.Exp(λ = 0.01, γ = 0.5)
ParameterSchedulers.Exp{Float64}(0.01, 0.5)

julia> scheduler = Schedule.Scheduler(schedule, opt)
Scheduler(ParameterSchedulers.Exp{Float64}(0.01, 0.5), Momentum(0.01, 0.9, IdDict{Any,Any}()))
"""
struct Scheduler{T<:AbstractSchedule, O}
  schedule::T
  optimiser::O
  iter::IdDict{Any, Int}
end
Scheduler(schedule, optimiser) = Scheduler(schedule, optimiser, IdDict{Any, Int}())

Base.show(io::IO, o::Scheduler) =
  print(io, "Scheduler(", o.schedule, ", ", o.optimiser, ")")

function Optimise.apply!(opt::Scheduler, x, dx)
  # set param
  t = get!(opt.iter, x, 1)
  opt.optimiser.eta = opt.schedule[t]
  opt.iter[x] += 1

  # do normal apply
  return Optimise.apply!(opt.optimiser, x, dx)
end