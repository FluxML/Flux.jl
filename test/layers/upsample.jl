@testset "bilinear upsample" begin
  m = Upsample(scale=(2, 3), mode=:bilinear)
  y = m((rand(Float32, 3, 4, 2, 3)))
  @test y isa Array{Float32, 4} 
  @test size(y) == (6, 12, 2, 3)
end

@testset "PixelShuffle" begin
  m = PixelShuffle(3)
  y = m((rand(Float32, 3, 4, 18, 3)))
  @test y isa Array{Float32, 4} 
  @test size(y) == (9, 12, 2, 3)
end
