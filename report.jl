using Enzyme
import Flux

struct Dense{Tw<:AbstractArray, Tb}
    weight::Tw
    bias::Tb
end

function size_check(layer, x::AbstractArray, (d, n)::Pair)
    size(x, d) == n || throw(DimensionMismatch(" expects size(input, $d) == $n, but got "))
end

function (a::Dense)(x::AbstractVecOrMat)
    size_check(a, x, 1 => size(a.weight, 2))
    return a.weight * x .+ a.bias
end


x = randn(2)
loss(model) = sum(model(x))

@info "autodiff1"

model = Dense(randn(2,2), zeros(2))
dmodel = Enzyme.make_zero(model)
autodiff(ReverseWithPrimal, loss, Active, Duplicated(model, dmodel))

# @info "autodiff2"

# model = Flux.Dense(randn(2,2), zeros(2))
# dmodel = Enzyme.make_zero(model)
# autodiff(ReverseWithPrimal, loss, Active, Duplicated(model, dmodel))