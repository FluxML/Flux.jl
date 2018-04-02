using Base.Test

@testset "losses" begin
  @testset "mse" begin
    ŷ, y = randn(3,3), randn(3,3)
    @test mse(ŷ, y, average=false) ≈ sum((y.-ŷ).^2)
    @test mse(ŷ, y) ≈ sum((y.-ŷ).^2) / 3
  end
  
  @testset "cross entropy" begin
    # todo: move to util.jl?
    function onehot!(ŷ::AbstractMatrix, y::AbstractVector{T}) where T<:Integer
      ŷ .= 0
      for (j, i) in enumerate(y)
        ŷ[i,j] = 1
      end
      ŷ
    end

    ŷ = randn(3,5)
    ŷsoft = logsoftmax(ŷ)
    y = rand(1:3, 5)
    yonehot = onehot!(similar(ŷ), y) 
    weight = rand(3)

    @test nll(ŷsoft, y, reduce=false) ≈ sum(-yonehot .* ŷsoft, 1) |> vec
    @test nll(ŷsoft, y, reduce=false, weight=weight) ≈ sum(-yonehot .* ŷsoft .* weight, 1) |> vec
    @test nll(ŷsoft, y, average=false) ≈ sum(@. -yonehot * ŷsoft)
    @test nll(ŷsoft, y) ≈ sum(@. -yonehot * ŷsoft) / size(ŷ, 2)
    
    @test cross_entropy(ŷ, y, reduce=false) ≈ nll(ŷsoft, y, reduce=false) 
    @test cross_entropy(ŷ, y, average=false) ≈ nll(ŷsoft, y, average=false)
    @test cross_entropy(ŷ, y) ≈ nll(ŷsoft, y)
  end

  @testset "binary cross entropy" begin
    for (ŷ, y) in [(randn(3), rand(3)), (randn(3,4), rand(3,4)), (randn(3,4,5), rand(3,4,5))]
      @test bce(σ.(ŷ), y, reduce=false) ≈ @. -y*log(σ(ŷ)) - (1 - y)*log(1 - σ(ŷ))
      @test bce(σ.(ŷ), y, average=false) ≈ sum(@. -y*log(σ(ŷ)) - (1 - y)*log(1 - σ(ŷ)))
      @test bce(σ.(ŷ), y) ≈ sum(@. -y*log(σ(ŷ)) - (1 - y)*log(1 - σ(ŷ))) / size(ŷ, ndims(ŷ))
      
      @test bce_logit(ŷ, y, reduce=false) ≈ bce(σ.(ŷ), y, reduce=false)
      @test bce_logit(ŷ, y, average=false) ≈ bce(σ.(ŷ), y, average=false)
      @test bce_logit(ŷ, y) ≈ bce(σ.(ŷ), y)
    end
  end

end
  
