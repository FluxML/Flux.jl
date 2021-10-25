for n in [2, 20, 200, 2000]
    x = Flux.onehotbatch(1:n, 1:n)
    model = Dense(n, n)
    println("CPU n=$n")
    run_benchmark(model, x, cuda=false)
    println("CUDA n=$n")
    run_benchmark(model, x, cuda=true)    
end
