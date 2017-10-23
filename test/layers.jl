@testset "dropout" begin
  x = [1.,2.,3.]
  @test x === Dropout(0.1, mode=:eval)(x)
  @test x === Dropout(0, mode=:train)(x)
  @test all(zeros(x) .== Dropout(1, mode=:train)(x))

  x = rand(100)
  m = Dropout(0.9)
  y = m(x)
  @test count(a->a==0, y) > 50
  setmode!(m, :eval)
  y = m(x)
  @test count(a->a==0, y) == 0

  x = rand(100)
  m = Chain(Dense(100,100),
            Dropout(0.9))
  y = m(x)
  @test count(a->a.data[] == 0, y) > 50
  setmode!(m, :eval)
  y = m(x)
  @test count(a->a.data[] == 0, y) == 0
end
