@net type TLP
  first
  second
  function (x)
    l1 = σ(first(x))
    l2 = softmax(second(l1))
  end
end

function test_tupleio(bk)
  @testset "Tuple I/O" begin
    val = [1,2,3]
    tup = ([1,2,3],[4,5,6])
    @test bk(@net x -> (identity(x),))(val) == (val,)
    @test bk(@net x -> x[1].*x[2])(tup) == [4,10,18]
  end
end

function test_recurrence(bk)
  @testset "Recurrence" begin
    seq = unsqueeze(stack(rand(10) for i = 1:3))
    r = unroll(Recurrent(10, 5), 3)
    rm = bk(r)
    @test r(seq) ≈ rm(seq)
  end
end

function test_back(bk)
  @testset "Backward Pass" begin
    xs, ys = rand(1, 20), rand(1, 20)
    d = Affine(20, 10)
    dm = bk(d)
    d′ = deepcopy(d)
    @test dm(xs) ≈ d(xs)
    @test dm(xs) ≈ d′(xs)

    Δ = back!(dm, randn(1, 10), xs)
    @test length(Δ[1]) == 20
    update!(dm, 0.1)

    @test dm(xs) ≈ d(xs)
    @test !(dm(xs) ≈ d′(xs))
  end
end

function test_stacktrace(bk)
  @testset "Stack Traces" begin
    model = TLP(Affine(10, 20), Affine(21, 15))
    dm = bk(model)
    e = try dm(rand(1, 10))
    catch e e end

    @test isa(e, DataFlow.Interpreter.Exception)
    @test e.trace[1].func == Symbol("Flux.Affine")
    @test e.trace[2].func == :TLP
  end
end

function test_anon(bk)
  @testset "Closures" begin
    x, y = rand(3), rand(5)
    model = bk(@net xs -> map(x -> x .* x, xs))
    @test all(model((x, y)) .≈ (x.*x, y.*y))
  end
end
