using Flux

function main()
    # x = rand(Float32, 20, 16)
    # d = Dense(20 => 40)
    x = rand(Float32, 128, 1, 16)
    d = Conv((3,), 1 => 2)

    @show size(d.weight)

    wn = Flux.WeightNorm(d, :weight)
    @show size(wn.g)
    y1 = wn(x)

    w = Flux.weightnorm(wn)
    y2 = w(x)
    @assert y1 ≈ y2
    return
end
main()
