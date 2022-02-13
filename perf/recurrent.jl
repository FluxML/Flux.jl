
println("RNN")
for n in [2, 20, 200, 2000], T in [1, 8, 16, 64]
  x = [randn(Float32, n, n) for t in 1:T]
  model = RNN(n, n)
  println("CPU n=$n, t=$T")
  run_benchmark(model, x, cuda=false)
  println("CUDA n=$n, t=$T")
  run_benchmark(model, x, cuda=true)    
end

println("RNN-3d")
for n in [2, 20, 200, 2000], T in [1, 8, 16, 64]
  x = randn(Float32, n, n, T)
  model = RNN(n, n)
  println("CPU n=$n, t=$T")
  run_benchmark(model, x, cuda=false)
  println("CUDA n=$n, t=$T")
  run_benchmark(model, x, cuda=true)    
end



