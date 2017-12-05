using Flux: testmode!

@testset "Dropout" begin
  x = [1.,2.,3.]
  @test x == testmode!(Dropout(0.1))(x)
  @test x == Dropout(0)(x)
  @test zeros(x) == Dropout(1)(x)

  x = rand(100)
  m = Dropout(0.9)
  y = m(x)
  @test count(a->a==0, y) > 50
  testmode!(m)
  y = m(x)
  @test count(a->a==0, y) == 0
  testmode!(m, false)
  y = m(x)
  @test count(a->a==0, y) > 50

  x = rand(100)
  m = Chain(Dense(100,100),
            Dropout(0.9))
  y = m(x)
  @test count(a->a == 0, y) > 50
  testmode!(m)
  y = m(x)
  @test count(a->a == 0, y) == 0
end
