
println("RNN")
for n in [2, 20, 200, 1000], T in [1, 8, 16, 64]
  x = [randn(Float32, n, n) for t in 1:T]
  model = RNN(n, n)
  println("CPU n=$n, t=$T")
  run_benchmark(model, x, cuda=false)
  println("CUDA n=$n, t=$T")
  try
      run_benchmark(model, x, cuda=true)
  catch ex
      @show typeof(ex)
      if ex isa OutOfGPUMemoryError
          @warn "Not enough GPU memory to run test"
      else
          rethrow(ex)
      end
  end
end

println("RNN-3d")
for n in [2, 20, 200, 1000], T in [1, 8, 16, 64]
  x = randn(Float32, n, n, T)
  model = RNN(n, n)
  println("CPU n=$n, t=$T")
  run_benchmark(model, x, cuda=false)
  println("CUDA n=$n, t=$T")
  try
      run_benchmark(model, x, cuda=true)
  catch ex
      @show typeof(ex)
      if ex isa OutOfGPUMemoryError
          @warn "Not enough GPU memory to run test"
      else
          rethrow(ex)
      end
  end
end



