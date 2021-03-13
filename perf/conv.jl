for ch in [1, 3, 16, 64]
    x = rand(64, 64, ch, 64)
    model = Conv((3,3), ch=>ch)
    println("CPU ch=$ch")
    run_benchmark(model, x, cuda=false)
    println("CUDA ch=$ch")
    run_benchmark(model, x, cuda=true)    
end
