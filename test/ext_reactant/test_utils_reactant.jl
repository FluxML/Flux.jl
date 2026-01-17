# These are used only in test_utils.jl but cannot leave there 
# because Reactant is only optionally loaded and the macros fail when it is not loaded.

function reactant_withgradient(f, x...)
    y, g = Reactant.@jit Flux.withgradient(f, AutoEnzyme(), x...)
    return y, g
end

function reactant_loss(loss, x...)
    l = Reactant.@jit loss(x...)
    @test l isa Reactant.ConcreteRNumber
    return l
end
