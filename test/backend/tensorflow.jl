using TensorFlow
Flux.loadtf()

@testset "TensorFlow" begin

xs, ys = rand(1, 20), rand(1, 20)
d = Affine(20, 10)

dt = tf(d)
@test d(xs) ≈ dt(xs)

test_tupleio(tf)
test_recurrence(tf)
test_stacktrace(tf)

@testset "Tensor interface" begin
  sess = TensorFlow.Session()
  X = placeholder(Float32)
  Y = Tensor(d, X)
  run(sess, global_variables_initializer())

  @test run(sess, Y, Dict(X=>xs)) ≈ d(xs)
end

@testset "Ops" begin

error_margin = 1e-6
#using Flux,Base.Test

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# reshape
xs = convert(Array{Float32},randn(6,3))
@net f(x,y) = reshape(x,y)
m = tf(f)
@test maximum(abs(reshape(xs,(9,2)) - transpose(m(transpose(xs),[2,9])))) < error_margin # Note: TF is row major and julia is not

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# transpose
@net f(x) = transpose(x)
m = tf(f)
@test maximum(abs(m(xs)-transpose(xs))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# permutedims
xs = convert(Array{Float32},randn(6,3,2))
@net f(x,y) = permutedims(x,y)
m = tf(f)
@test maximum(abs(m(xs,[3,2,1])-permutedims(xs,[3,2,1]))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# tile
xs = convert(Array{Float32},randn(4,3,1))
@net f(x,y) = tile(x,y)
m = tf(f)
@test maximum(abs(m(xs,[1,1,3])-repeat(xs,outer=(1,1,3)))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# chol
z = convert(Array{Float32},randn(6,5))
z = z'*z
@net f(x) = chol(x)
m = tf(f)
@test maximum(abs(m(z)-chol(z))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# size
z = randn(4,5)
@net f(x) = size(x)
m = tf(f)
@test maximum(abs(m(z)-[size(z)...])) == 0

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# cat
z1 = convert(Array{Float32},randn(4,1))
z2 = convert(Array{Float32},randn(4,1))
@net f(x,y) = cat(2,x,y)
m = tf(f)
@test maximum(abs(m(z1,z2)-cat(2,z1,z2))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

end
