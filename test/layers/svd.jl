using Flux
using Flux: svd_back
using Flux.Tracker: @grad, data, track, TrackedTuple
using Test

@testset "svdbp" begin
    M, N = 4, 6
    K = min(M, N)
    A = randn(M, N)
    U, S, V = svd(A)
    dU, dS, dV = randn(M, K), randn(K), randn(N, K)
    dA = svd_back(U, S, V, dU, dS, dV)

    for i in 1:length(A)
        δ = 0.01
        A[i] -= δ/2
        U1, S1, V1 = svd(A)
        A[i] += δ
        U2, S2, V2 = svd(A)
        A[i] -= δ/2
        δS = S2 .- S1
        δU = U2 .- U1
        δV = V2 .- V1
        @test isapprox(sum(dS .* δS) + sum(dU .* δU) + sum(dV .* δV), dA[i] * δ, atol=1e-5)
    end
end

@testset "svdflux" begin
    M, N = 4, 6
    K = min(M, N)
    A = randn(M, N)
    PA = A|>param
    res = svd(PA)
    U, S, V = res
    dU, dS, dV = randn(M, K), randn(K), randn(N, K)
    Tracker.back!(res, (dU, dS, dV))
    dA = Tracker.grad(PA)

    for i in 1:length(A)
        δ = 0.01
        A[i] -= δ/2
        U1, S1, V1 = svd(A)
        A[i] += δ
        U2, S2, V2 = svd(A)
        A[i] -= δ/2
        δS = S2 .- S1
        δU = U2 .- U1
        δV = V2 .- V1
        @test isapprox(sum(dS .* δS) + sum(dU .* δU) + sum(dV .* δV), dA[i] * δ, atol=1e-5)
    end
end
