export SGD

struct Optimizer
    cache::Dict{Param, Vector{Function}}
    steps
    Optimizer(steps) = new(Dict{Param, Function}(), steps)
end

function update!(p::Param, o::Optimizer)
    steps = Base.@get!(o.cache, p, map(x->x(p), o.steps))
    foreach(f->f(p), steps)
    p.x .-= p.Δx
end

function Momentum(η)
    function (p)
        momentum = zeros(p.x)
        
        function (p)
            momentum .= η .* momentum .+ p.Δx
            p.Δx .= momentum
        end
    end
end

function NesterovMomentum()
    error("TODO")
end

function WeightDecayConst()
    error("TODO")
end

function WeightDecayRatio()
    error("TODO")
end

function GradDecayFix(lr)
    function (p::Param)
        function (p::Param)
            p.Δx .= lr .* p.Δx
        end
    end
end

function GradDecayExp()
    error("TODO")
end

function GradDecayInv()
    error("TODO")
end

function WeightClipConst()
    error("TODO")
end

function WeightClipNorm()
    error("TODO")
end

function GradClipConst()
    error("TODO")
end

function GradClipNorm()
    error("TODO")
end

macro restrict_range(var::Symbol, range::String)
    left, right = split(range, ", ")
    lo = left[1] == '[' ? :>= : :>
    lt = left[2:end]
    ro = right[end] == ']' ? :<= : :<
    rt = right[1:end-1]

    error_msg = "$var ∈ $range must be hold"
    var = esc(var)

    quote
        $( lt != "-∞" && :( $lo($var, $(parse(Float64, lt))) || throw(ArgumentError($error_msg)) ) )
        $( rt != "∞"  && :( $ro($var, $(parse(Float64, rt))) || throw(ArgumentError($error_msg)) ) )
    end
end

"""
Stochastic gradient descent optimizer.
Includes support for momentum,
learning rate decay, and Nesterov momentum.

# Arguments
    lr: float >= 0. Learning rate.
    momentum: float >= 0. Parameter updates momentum.
    decay: float >= 0. Learning rate decay over each update.
    nesterov: boolean. Whether to apply Nesterov momentum.
"""
function SGD(; lr::Real=.1,
               momentum::Real=0,
               decay::Real=0,
               nesterov::Bool=false)

    @restrict_range lr       "[0, ∞)"
    @restrict_range momentum "[0, 1]"
    @restrict_range decay    "[0, ∞)"

    steps = []

    if momentum != 0
        nesterov ? push!(steps, NesterovMomentum(momentum)) :
                   push!(steps, Momentum(momentum))
    end

    decay != 0 && push!(steps, GradDecayInv(decay))

    lr != 1 && push!(steps, GradDecayFix(lr))

    Optimizer(steps)
end