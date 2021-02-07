@testset "upsample bilinear" begin
  m = Upsample((2, 3), mode=:bilinear)
  x = rand(Float32, 3, 4, 2, 3)
  y = m(x)
  @test y isa Array{Float32, 4} 
  @test size(y) == (6, 12, 2, 3)

  m = Upsample(3, mode=:bilinear)
  x = rand(Float32, 3, 4, 2, 3)
  y = m(x)
  @test y isa Array{Float32, 4} 
  @test size(y) == (9, 12, 2, 3)
end

@testset "upsample nearest" begin
  x = rand(Float32, 3, 2, 3)
  m = Upsample((2,), mode=:nearest)
  y = m(x)
  @test y isa Array{Float32, 3} 
  @test size(y) == (6, 2, 3)

  x = rand(Float32, 3, 4, 2, 3)
  
  m = Upsample((2, 3), mode=:nearest)
  y = m(x)
  @test y isa Array{Float32, 4} 
  @test size(y) == (6, 12, 2, 3)
  
  m = Upsample((2,), mode=:nearest)
  y = m(x)
  @test y isa Array{Float32, 4} 
  @test size(y) == (6, 4, 2, 3)

  m = Upsample(2, mode=:nearest)
  y = m(x)
  @test y isa Array{Float32, 4} 
  @test size(y) == (6, 8, 2, 3)
end

@testset "PixelShuffle" begin
  m = PixelShuffle(2)
  x = rand(Float32, 3, 18, 3)
  y = m(x)
  @test y isa Array{Float32, 3} 
  @test size(y) == (6, 9, 3)

  m = PixelShuffle(3)
  x = rand(Float32, 3, 4, 18, 3)
  y = m(x)
  @test y isa Array{Float32, 4} 
  @test size(y) == (9, 12, 2, 3)
end
