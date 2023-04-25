using Metalhead, Flux

m1 = ResNet(18)
m2 = ResNet(18)
@time Flux.loadmodel!(m2, m1) # warmup
@time Flux.loadmodel!(m2, m1)
#  0.003388 seconds (23.01 k allocations: 2.157 MiB) # this PR

##  SAVE AND LOAD
using Functors

function state(x)
    if Functors.isleaf(x)
        return x
    else
        return map(state, Functors.children(x))
    end
end

s = state(m1);
@time Flux.loadmodel!(m2, s); # warmup
@time Flux.loadmodel!(m2, s);
