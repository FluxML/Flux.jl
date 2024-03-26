for n in [2, 20, 200, 2000]
    x = randn(Float32, n, n)
    model = Dense(n => n)
    println("CPU n=$n")
    run_benchmark(model, x, cuda=false)
    println("CUDA n=$n")
    run_benchmark(model, x, cuda=true)    
end
