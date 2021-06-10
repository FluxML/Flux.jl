@testset "use params in gradient context"
    m = Chain(Dense(3,2), Dense(2,2))
    ps = Flux.params()
    gs = gradient(() -> sum(sum(p) for p in Flux.params(m)), ps)
    for p in ps
        @test gs[p] ≈ ones(size(p))
    end    
begin
