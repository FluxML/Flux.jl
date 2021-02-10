@testset "upsample bilinear" begin
  m = Upsample(:bilinear, scale=(2, 3))
  x = rand(Float32, 3, 4, 2, 3)
  y = m(x)
  @test y isa Array{Float32, 4} 
  @test size(y) == (6, 12, 2, 3)

  m = Upsample(:bilinear, scale=3)
  x = rand(Float32, 3, 4, 2, 3)
  y = m(x)
  @test y isa Array{Float32, 4} 
  @test size(y) == (9, 12, 2, 3)

  m = Upsample(:bilinear, size=(4, 6))
  x = rand(Float32, 3, 4, 2, 3)
  y = m(x)
  @test y isa Array{Float32, 4} 
  @test size(y) == (4, 6, 2, 3)
end

@testset "upsample nearest" begin
  x = rand(Float32, 3, 2, 3)
  m = Upsample(:nearest, scale=(2,))
  y = m(x)
  @test y isa Array{Float32, 3} 
  @test size(y) == (6, 2, 3)

  x = rand(Float32, 3, 4, 2, 3)
  
  m = Upsample(:nearest, scale=(2, 3))
  y = m(x)
  @test y isa Array{Float32, 4} 
  @test size(y) == (6, 12, 2, 3)
  
  m = Upsample(:nearest, scale=(2,))
  y = m(x)
  @test y isa Array{Float32, 4} 
  @test size(y) == (6, 4, 2, 3)

  m = Upsample(:nearest, scale=2)
  y = m(x)
  @test y isa Array{Float32, 4} 
  @test size(y) == (6, 8, 2, 3)

  m = Upsample(2)
  @test y2 ≈ y 

  m = Upsample(:nearest, size=(6,8))
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
