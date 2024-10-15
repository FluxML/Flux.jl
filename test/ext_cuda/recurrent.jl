@testset for R in (RNN,)
  m = R(3 => 5)
  x = randn(Float32, 3, 4)
  h = randn(Float32, 5)
  test_gradients(m, x, h, test_gpu=true, compare_finite_diff=false)
end
