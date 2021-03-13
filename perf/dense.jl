for n in [4, 16, 64, 256, 1024]
    x = rand(n, n)
    model = Dense(n, n)
    println("CPU n=$n")
    run_benchmark(model, x, cuda=false)
    println("CUDA n=$n")
    run_benchmark(model, x, cuda=true)    
end
