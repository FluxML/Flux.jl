using Flux: @treelike

# should this be HyperNet{H,M} ?
struct HyperNet
    h
    m
end

@treelike HyperNet

function (H::HyperNet)(t)
    _, re = destructure(H.m)
    ps = H.h(t)
    return re(ps)
end

function HyperDense(in_dim,m,σ)
    p,_ = destructure(m)
    return Dense(in_dim,length(p),σ)
end

HyperDense(in_dim,m) = HyperDense(in_dim,m,identity)
