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

error_margin = 1e-4
#using Flux,Base.Test

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# reshape
xs = convert(Array{Float32},randn(6,3))
@net f(x,y) = reshape(x,y)
m = tf(f)
@test maximum(abs.(reshape(xs,(9,2)) - transpose(m(transpose(xs),[2,9])))) < error_margin # Note: TF is row major and julia is not

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# transpose
@net f(x) = transpose(x)
m = tf(f)
@test maximum(abs.(m(xs)-transpose(xs))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# permutedims
xs = convert(Array{Float32},randn(6,3,2))
@net f(x,y) = permutedims(x,y)
m = tf(f)
@test maximum(abs.(m(xs,[3,2,1])-permutedims(xs,[3,2,1]))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# tile
xs = convert(Array{Float32},randn(4,3,1))
@net f(x,y) = Flux.tile(x,y)
m = tf(f)
@test maximum(abs.(m(xs,[1,1,3])-repeat(xs,outer=(1,1,3)))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# chol
z = convert(Array{Float32},randn(6,5))
z = z'*z
@net f(x) = chol(x)
m = tf(f)
@test maximum(abs.(m(z)-chol(z))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# size
z = randn(4,5)
@net f(x) = size(x)
m = tf(f)
@test maximum(abs.(m(z)-[size(z)...])) == 0

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# cat
z1 = convert(Array{Float32},randn(4,1))
z2 = convert(Array{Float32},randn(4,1))
@net f(x,y) = cat(2,x,y)
m = tf(f)
@test maximum(abs.(m(z1,z2)-cat(2,z1,z2))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# fill
@net f(x,y) = fill(x,y)
m = tf(f)
@test maximum(abs.(m(3.2,[3,2])-convert(Array{Float32},3.2*ones(3,2)))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# slice
@net f(a,b,c) = Flux.slice(a,b,c)
m = tf(f)
z = convert(Array{Float32},randn(6,8))
@test maximum(abs.(m(z,[3,4],[3,-1])-convert(Array{Float32},Flux.slice(z,[3,4],[3,-1])))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# length
@net f(x) = length(x)
m = tf(f)
@test m(z) == length(z)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# pad
@net f(x,y) = Flux.pad(x,y)
m = tf(f)
@test maximum(abs.(m(z,[3 4;1 2])-Flux.pad(z,[3 4;1 2]))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# cast
@net f(x) = Flux.cast(x,Int32)
m = tf(f)
z = zeros(4,3)
@test typeof(m(z)) == Matrix{Int32}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# randu
@net f(x) = randu(x)
m = tf(f)
y = m([2,3])
@test all(y .>= 0)
@test all(y .<= 1)
@test size(y) == (2,3)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# randn
@net f(x) = randn(x)
m = tf(f)
y = m([2,3])
@test size(y) == (2,3)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# solve
A = convert(Array{Float32},randn(5,5))
b = convert(Array{Float32},randn(5,1))
@net f(x,y) = solve(x,y)
m = tf(f)
@test maximum(abs.(m(A,b)-A\b)) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# triangular_solve
_,A,_ = lu(A)
@net f(x,y) = triangular_solve(x,y)
m = tf(f)
@test maximum(abs.(m(A,b)-A\b)) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# inv
A = convert(Array{Float32},randn(5,5))
@net f(x) = inv(x)
m = tf(f)
@test maximum(abs.(m(A)-inv(A))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# det
@net f(x) = det(x)
m = tf(f)
@test maximum(abs.(m(A)-det(A))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# diag
@net f(x) = diag(x)
m = tf(f)
@test maximum(abs.(m(A)-diag(A))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# diagm
z = convert(Array{Float32},randn(5))
@net f(x) = diagm(x)
m = tf(f)
@test maximum(abs.(m(z)-diagm(z))) < error_margin

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# svd
@net f(x) = svd(x)
m = tf(f)
s,u,v = m(A)
u2,s2,v2 = svd(A)
maximum(abs.(s-s2)) < error_margin
maximum(abs.(u*diagm(s)*transpose(v) - A)) < error_margin


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

end
