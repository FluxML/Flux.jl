using Flux, Functors, Optimisers

module MacroTest
  using Flux: @layer

  struct Duo{T,S}; x::T; y::S; end
  @layer :expand Duo

  struct Trio; a; b; c end
  # @layer Trio trainable=(a,b) test=(c) # should be (c,) but it lets you forget
  @layer Trio trainable=(a,b)  # defining a method for test is made an error, for now

  struct TwoThirds; a; b; c; end
end

@testset "@layer macro" begin
  @test !isdefined(MacroTest, :Flux)  # That's why the module, to check scope

  m2 = MacroTest.Duo(Dense(2=>2), Chain(Flux.Scale(2), Dropout(0.2)))

  @test Functors.children(m2) isa NamedTuple{(:x, :y)}
  @test length(Optimisers.destructure(m2)[1]) == 10

  m3 = MacroTest.Trio([1.0], [2.0], [3.0])

  @test Functors.children(m3) isa NamedTuple{(:a, :b, :c)}
  @test fmap(zero, m3) isa MacroTest.Trio

  @test Optimisers.trainable(m3) isa NamedTuple{(:a, :b)}
  @test Optimisers.destructure(m3)[1] == [1, 2]

  # @test MacroTest.test(m3) == (c = [3.0],)  # removed, for now
  
  m23 = MacroTest.TwoThirds([1 2], [3 4], [5 6])
  # Check that we can use the macro with a qualified type name, outside the defining module:
  Flux.@layer :expand MacroTest.TwoThirds children=(:a,:c) trainable=(:a)  # documented as (a,c) but allow quotes

  @test Functors.children(m23) == (a = [1 2], c = [5 6])
  m23re = Functors.functor(m23)[2]((a = [10 20], c = [50 60]))
  @test m23re isa MacroTest.TwoThirds
  @test Flux.namedtuple(m23re) == (a = [10 20], b = [3 4], c = [50 60])

  @test Optimisers.trainable(m23) == (a = [1 2],)
end

