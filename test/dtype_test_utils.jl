all_dtype_equal(params, T) = all(eltype(eltype(p)) == T for p in params)

"""
    test_layer_dtype(treelike_factory)

Takes a method that takes in `nothing`, `Float32`, or `Float64` and returns a
treelike type, such as Dense.
Ensures that the treelike's params are of the desired type.
"""
function test_layer_dtype(treelike_factory)
  @test all_dtype_equal(params(treelike_factory(nothing)), Flux.FloatX)
  for T in [Float32, Float64]
    @test all_dtype_equal(params(treelike_factory(T)), T)
  end
end
