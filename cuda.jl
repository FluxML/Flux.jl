using Flux, CUDA

BN = BatchNorm(3) |> gpu;
x = randn(2, 2, 3, 4) |> gpu;

NNlib.batchnorm(BN.γ, BN.β, x, BN.μ, BN.σ², BN.momentum; 
                  alpha=1, beta=0, eps=BN.ϵ, 
                  training=Flux._isactive(BN, x))

