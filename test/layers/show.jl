
@testset "layer printing" begin # 2-arg show, defined with layes

  @test repr(Dense(2,3)) == "Dense(2 => 3)"
  @test repr(Chain(Dense(2,3))) == "Chain(Dense(2 => 3))"
  @test repr(Chain(lay=Dense(2,3))) == "Chain(lay = Dense(2 => 3))"

end
@testset "nested model printing" begin # 3-arg show, defined in show.jl

  # Dense -- has parameter count, but not when inside a matrix:

  toplevel_dense = repr("text/plain", Dense(2,3))
  @test occursin("Dense(2 => 3)", toplevel_dense)
  @test occursin("# 9 parameters", toplevel_dense)

  @test Meta.isexpr(Meta.parse(toplevel_dense), :call)  # comment is ignored

  vector_dense = repr("text/plain", [Dense(2,3), Dense(2,3)])
  @test occursin("Dense(2 => 3)", vector_dense)
  @test occursin("# 9 parameters", vector_dense)

  matrix_dense = repr("text/plain", fill(Dense(2,3), 3, 3))
  @test occursin("Dense(2 => 3)", matrix_dense)
  @test !occursin("# 9 parameters", matrix_dense)

  tuple_dense = repr("text/plain", tuple(Dense(2,3)))
  @test occursin("Dense(2 => 3)", tuple_dense)
  @test !occursin("# 9 parameters", tuple_dense)

  # Chain -- gets split over lines at top level only

  toplevel_chain = repr("text/plain", Chain(Dense(2,3)))
  @test occursin("Chain(\n  Dense(2 => 3)", toplevel_chain)
  @test occursin("# 9 parameters", toplevel_chain)
  @test !occursin("# Total:", toplevel_chain)

  vector_chain = repr("text/plain", [Chain(Dense(2,3)), Chain(Dense(2,3))])
  @test occursin("Chain(Dense(2 => 3))", vector_chain)
  @test occursin("# 9 parameters", vector_chain)
  @test !occursin("# Total:", vector_chain)

  matrix_chain = repr("text/plain", fill(Chain(Dense(2,3)), 3,3))
  @test occursin("Chain(Dense(2 => 3))", matrix_chain)
  @test !occursin("# 9 parameters", matrix_chain)
  @test !occursin("# Total:", matrix_chain)

  # ... and only long enough chains get a total at the end:

  longchain = Chain(Dense(2 => 3), Dense(3 => 4), Dense(4 => 5), softmax)

  toplevel_longchain = repr("text/plain", longchain)
  @test occursin("Chain(\n  Dense(2 => 3)", toplevel_longchain)
  @test occursin("# 9 parameters", toplevel_longchain)
  @test occursin("# Total: 6 arrays, 50 parameters", toplevel_longchain)

  vector_longchain = repr("text/plain", [longchain, longchain]) # pretty ugly in reality
  @test occursin("Chain(Dense(2 => 3)", vector_longchain)
  @test occursin("# 50 parameters", vector_longchain)
  @test !occursin("# 9 parameters", vector_longchain)
  @test !occursin("# Total:", vector_longchain)

  matrix_longchain = repr("text/plain", fill(longchain, 3,3))
  @test occursin("Chain(Dense(2 => 3)", matrix_longchain)
  @test !occursin("# 9 parameters", matrix_longchain)
  @test !occursin("# Total:", matrix_longchain)

  @test Meta.isexpr(Meta.parse(toplevel_longchain), :call)  # comments are ignored
  @test Meta.parse(toplevel_longchain).args[1] == :Chain

  # Functors@0.3 marks transposed matrices non-leaf, shouldn't affect printing:
  adjoint_chain = repr("text/plain", Chain([Dense([1 2; 3 4]')]))
  @test occursin("Dense(2 => 2)", adjoint_chain)
  @test occursin("Chain(", adjoint_chain)
  @test occursin("[", adjoint_chain)

  # New printing of arrays, and Fix1
  fix_chain = repr("text/plain", Chain(Base.Fix1(*, rand32(22,33)), softmax))
  @test occursin("Fix1(", fix_chain)
  @test occursin("22×33 Matrix{Float32}", fix_chain)
end

# Bug when no children, https://github.com/FluxML/Flux.jl/issues/2208
struct NoFields end
Flux.@layer NoFields

@testset "show with no fields" begin
  str = repr("text/plain", Chain(Dense(1=>1), Dense(1=>1), NoFields()))
  @test occursin("4 arrays, 4 parameters", str)
end

# @layer :named shows fieldnames in expanded output
struct NamedContainer; inner; extra; end
Flux.@layer :named NamedContainer

@testset "named layer printing" begin
  # Custom type with :named
  nc = NamedContainer(Dense(2, 3), Dense(3, 4))
  str = repr("text/plain", nc)
  @test occursin("inner = Dense(2 => 3)", str)
  @test occursin("extra = Dense(3 => 4)", str)


end

# The printing code reads the fields of layers directly, when `Functors.functor` and
# `Optimisers.trainable` are the default methods, and must notice when they are not:
struct OnlyA; a; b; end
Flux.@layer OnlyA
Functors.@functor OnlyA (a,)

struct TrainA; a; b; end
Optimisers.trainable(x::TrainA) = (; a = x.a)

struct LeafLayer; w; end
Functors.@leaf LeafLayer
Base.show(io::IO, ::LeafLayer) = print(io, "LeafLayer()")

@testset "custom functor, trainable, and leaf" begin
  str = repr("text/plain", Chain(OnlyA(rand32(2), rand32(3)), Dense(2=>2)))
  @test occursin(r"OnlyA\(.*# 2 parameters\n", str)  # b is not a child at all

  str = repr("text/plain", Chain(TrainA(rand32(2), rand32(3)), Dense(2=>2)))
  @test occursin(r"TrainA\(.*# 2 parameters, plus 3\n", str)

  str = repr("text/plain", Chain(LeafLayer(rand32(4)), Dense(2=>2), Dense(2=>2)))
  @test occursin("LeafLayer(),\n", str)
  @test occursin("Total: 4 arrays, 12 parameters", str)

  # Tied weights are counted once:
  d = Dense(3=>3)
  str = repr("text/plain", Chain(d, d, Dense(d.weight', false), Dense(3=>2)))
  @test occursin(r"Total: 4 [a-z ]*arrays, 20 parameters", str)
end

# Methods defined after a layer has been printed must be noticed the next time:
struct LateTrain; a; b; end
Flux.@layer LateTrain
late_before = repr("text/plain", LateTrain(rand32(2), rand32(3)))
Optimisers.trainable(x::LateTrain) = (; a = x.a)
late_after = repr("text/plain", LateTrain(rand32(2), rand32(3)))

@testset "methods defined after printing" begin
  @test occursin("# 5 parameters", late_before)
  @test occursin("# 2 parameters, plus 3 non-trainable", late_after)
end

@testset "IOContext properties are kept" begin
  m = Chain(Dense(2=>3), Dense(3=>3), Dense(3=>3))
  @test occursin("\e[90m# 9 parameters", sprint(show, MIME"text/plain"(), m; context = :color => true))
  @test !occursin("\e[", sprint(show, MIME"text/plain"(), m; context = :color => false))
end

@testset "all zero, some NaN, some Inf" begin
  shows(m) = repr("text/plain", m)
  @test !occursin(r"all zero|NaN|Inf", shows(Dense(2=>3)))
  @test occursin("(all zero)", shows(Dense(zeros32(2,3), false)))
  @test occursin("(some NaN)", shows(Dense(fill(NaN32, 2, 3))))
  @test occursin("(some Inf)", shows(Dense(fill(-Inf32, 2, 3))))
  @test occursin("(some NaN)", shows(Dense([Inf32 NaN32])))  # NaN wins
  @test occursin("(some Inf)", shows(Dense(Float16[1 Inf])))
  @test occursin("(some NaN)", shows(Dense(ComplexF32[1 NaN])))
  @test occursin("(some NaN)", shows(Conv(fill(NaN32, 3, 3, 1, 1), zeros32(1))))
  @test occursin("(some Inf)", shows(Dense(view(fill(Inf32, 4, 4), 1:2, 1:2))))  # not a DenseArray
  @test occursin("(some NaN)", shows(Chain(Dense(2=>2), Dense(fill(NaN32, 2, 2)), Dense(2=>2))))
  @test occursin("(some Inf)", shows(Dense(Real[1 Inf])))  # abstract eltype
  @test occursin("(all zero)", shows(Dense(Real[0 0], false)))
end

# A 1-arg `functor` method, as Flux's recurrent layers have, also replaces the fields:
struct OneArg; a; b; end
Flux.@layer OneArg
Functors.functor(x::OneArg) = (; a = x.a), y -> OneArg(y.a, x.b)

# An old-style `trainable` returning a Tuple selects children by `==`, as in Optimisers:
struct TupTrain; a; b; end
Flux.@layer TupTrain
Optimisers.trainable(x::TupTrain) = (x.a,)

@testset "1-arg functor, Tuple trainable" begin
  str = repr("text/plain", OneArg(rand32(2), rand32(3)))
  @test occursin("# 2 parameters", str)
  @test !occursin("non-trainable", str)

  Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
    v = rand32(2)
    @test occursin("# 4 parameters\n", repr("text/plain", Chain(TupTrain(v, copy(v)))))
    @test occursin("# 2 parameters, plus 2", repr("text/plain", Chain(TupTrain(v, rand32(2)))))
  end
end

struct LateFunctor; a; b; end
Flux.@layer LateFunctor
struct LateLeaf; w; end
late_functor_before = repr("text/plain", LateFunctor(rand32(2), rand32(3)))
late_leaf_before = repr("text/plain", Chain(LateLeaf(rand32(4)), Dense(2=>2)))
Functors.@functor LateFunctor (a,)
Functors.@leaf LateLeaf
late_functor_after = repr("text/plain", LateFunctor(rand32(2), rand32(3)))
late_leaf_after = repr("text/plain", Chain(LateLeaf(rand32(4)), Dense(2=>2)))

@testset "functor and leaf defined after printing" begin
  @test occursin("# 5 parameters", late_functor_before)
  @test occursin("# 2 parameters", late_functor_after)
  @test !occursin("non-trainable", late_functor_after)
  @test occursin("Total: 3 arrays, 10 parameters", late_leaf_before)
  @test !occursin("Total", late_leaf_after)  # only Dense has arrays now
end

# The shortcuts in show.jl must agree with Functors and Optimisers, for every node of these:
@testset "agrees with Functors and Optimisers" begin
  d = Dense(3=>3)
  models = Any[
    Chain(Dense(2=>3, relu), BatchNorm(3), Dropout(0.1), LayerNorm(3), Dense(3=>2)),
    Chain(Conv((3,3), 1=>2), MaxPool((2,2)), GroupNorm(2, 1), Flux.flatten),
    Chain(enc = Dense(2=>2), dec = Chain(Dense(2=>2), Dense(2=>1))),
    Chain([Dense(2=>2), Dense(2=>2)]),
    SkipConnection(Dense(2=>2), +), Parallel(vcat; a = Dense(2=>2), b = Dense(2=>3)),
    PairwiseFusion(+, Dense(2=>2), Dense(2=>2)), Maxout(() -> Dense(2=>2), 3),
    Embedding(5=>2), Flux.Bilinear((2,2)=>3), Flux.Scale(3), MultiHeadAttention(4; nheads=2),
    RNN(2=>3), LSTM(2=>3), GRU(2=>3), Chain(d, d, Dense(d.weight', false)),
    OnlyA(rand32(2), rand32(3)), TrainA(rand32(2), rand32(3)), LeafLayer(rand32(4)),
    OneArg(rand32(2), rand32(3)), LateFunctor(rand32(2), rand32(3)),
    (Dense(2=>2), (a = Dense(2=>2), b = [Dense(2=>2)])),
  ]
  fields(c) = c isa Flux._ShowFields ? (c.names, c.vals) :
    c isa NamedTuple ? (collect(keys(c)), collect(Any, values(c))) : (nothing, Flux._childvec(c))
  same(a, b) = fields(a)[1] == fields(b)[1] && all(map(===, fields(a)[2], fields(b)[2])) &&
    length(fields(a)[2]) == length(fields(b)[2])
  function check(x)
    @test Flux._show_isleaf(x) == Functors.isleaf(x)
    Functors.isleaf(x) && return
    @test same(Flux._show_functor_children(x), Functors.children(x))
    @test same(Flux._show_trainable(x), Optimisers.trainable(x))
    foreach(check, Flux._childvec(Functors.children(x)))
  end
  for m in models
    check(m)
    ps, qs = Flux._show_trainables(m), Optimisers.trainables(m)
    @test length(ps) == length(qs) && all(map(===, ps, qs))
  end
end

# Printing a new layer type must not compile any of the printing code again:
@testset "compiled once" begin
  for f in (Flux._layer_show, Flux._show_trainables!, Flux._show_arrays!, Flux._nan_show,
            Flux._show_functor_children, Flux._show_isleaf, Flux._show_trainable, Flux._show_methods)
    for m in methods(f)
      @test length(Base.specializations(m)) <= 1
    end
  end
  @test length(Base.specializations(which(Flux._big_show, Tuple{IO, Any, Int, Any}))) <= 1
end
