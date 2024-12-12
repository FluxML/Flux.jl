using Flux
using Zygote

function main()
    x = rand(Float32, 20, 1)
    c = Dense(20 => 20)

    # x = rand(Float32, 12, 1, 1)
    # c = Conv((3,), 1 => 2)
    y1 = c(x)
    wn = WeightNorm(c, :weight)
    @show wn
    y2 = wn(x)

    @assert y1 ≈ y2

    g = Zygote.gradient(wn) do wn
        sum(wn(x))
    end
    display(g); println()

    model = Chain(
        WeightNorm(Conv((3,), 1 => 2), :weight),
        WeightNorm(Conv((3,), 2 => 2), :weight),
    )
    @show model
    # y1 = model(x)

    mm = Flux.remove_weight_norms(model)
    @show mm
    return
end
main()
