
using Statistics
using Flux
using Test

#initial x value
# x = randn32(1000,1);
# x = [1,2,3,4,5]

# Mean
# E(xd + alpha(1-d)) = qu + (1-q)alpha
a_ = -1.7580993408473766
d = 0.2
q = 0.2
u = mean(x)  

function mean_test(x)
    # LHS
    mean_left = (x*d) .+ (a_*(1-d))
    mean_left = mean(mean_left)
    # println(mean_left)

    # RHS
    mean_right = (q*u) .+ ((1-q)*a_)
    # println(mean_right)
    @test isapprox(mean_left, mean_right, atol=0.2)
end

x = randn(2000,1);
@testset "Alphadropout Tests" begin
    mean_test(x);
end


# Variance
# Var(xd + alpha(1-d)) = q((1-q)(alpha-u)^2 + v)
# v = var(x)

# var_left = (x*d) .+ a_*(1-d)
# var_left = var(var_left)

# var_right = q*((1-q)*(a_-u).^2 + v)

# @test isapprox(var_left, var_right, atol=0.1)
