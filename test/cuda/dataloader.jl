using Flux, CUDA, Test
CUDA.allowscalar(false)

@testset "Flux.DeviceIterator" begin
  # Adapted from https://github.com/JuliaGPU/CUDA.jl/blob/master/test/iterator.jl
  batch_count = 10
  max_batch_items = 3
  max_ndims = 3
  sizes = 20:50
  rand_shape = () -> rand(sizes, rand(1:max_ndims))
  batches = [[rand(Float32, rand_shape()...) for _ in 1:rand(1:max_batch_items)]
                                     for _ in 1:batch_count]

  cubatches = Flux.DeviceIterator(Flux.FluxCUDAAdaptor(), batch for batch in batches) # ensure generators are accepted
  global previous_cubatch = missing
  for (batch, cubatch) in zip(batches, cubatches)
    global previous_cubatch
    @test ismissing(previous_cubatch) || all(x -> x.storage === nothing, previous_cubatch)
    @test batch == Array.(cubatch)
    @test all(x -> x isa CuArray, cubatch)
    previous_cubatch = cubatch
  end

  @test Base.IteratorSize(typeof(cubatches)) isa Base.HasShape{1}
  @test length(cubatches) == length(batch for batch in batches)
  @test axes(cubatches) == axes(batch for batch in batches)
  @test Base.IteratorEltype(typeof(cubatches)) isa Base.EltypeUnknown
  @test eltype(cubatches) == eltype(batch for batch in batches) == Any
  @test Base.IteratorEltype(typeof(Flux.DeviceIterator(Flux.FluxCUDAAdaptor(), batches))) isa Base.HasEltype
  @test eltype(Flux.DeviceIterator(Flux.FluxCUDAAdaptor(), batches)) == eltype(batches)  # Vector

  # Also check recursion into NamedTuple, and that it converts to Float32
  it_nt = Flux.DeviceIterator(Flux.FluxCUDAAdaptor(), (x=Float64[i,i/2], y=i) for i in 1:4)
  @test first(it_nt).x isa CuArray{Float32}
  batch1, state = iterate(it_nt)
  @test batch1.x == cu([1,1/2])
  batch2, _ = iterate(it_nt, state)
  @test batch2.x == cu([2,2/2])
  @test batch1.x.storage === nothing  # unsafe_free! has worked inside

  it_vec = Flux.DeviceIterator(Flux.FluxCUDAAdaptor(), [[i,i/2], [i/3, i/4]] for i in 1:4)
  @test first(it_vec)[1] isa CuArray{Float32}
end

# This is the only documented way to use DeviceIterator
@testset "gpu(::DataLoader)" begin
  X = randn(Float64, 3, 33)
  pre1 = Flux.DataLoader(X |> gpu; batchsize=13, shuffle=false)
  post1 = Flux.DataLoader(X; batchsize=13, shuffle=false) |> gpu
  @test pre1 isa Flux.DataLoader
  p1 = first(pre1)
  @test post1 isa Flux.DeviceIterator
  q1 = first(post1)
  for epoch in 1:2
    for (p, q) in zip(pre1, post1)
      @test p isa CuArray{Float32}
      @test q isa CuArray{Float32}
      @test p ≈ q
    end
  end
  @test p1.storage !== nothing
  @test q1.storage === nothing  # this memory has been freed

  Y = Flux.onehotbatch(rand(0:2, 33), 0:2)
  pre2 = Flux.DataLoader((x=X, y=Y) |> gpu; batchsize=7, shuffle=false)
  post2 = Flux.DataLoader((x=X, y=Y); batchsize=7, shuffle=false) |> gpu
  q2 = first(post2)
  for (p, q) in zip(pre2, post2)
    @test p.x == q.x
    @test_skip p.y == q.y  # https://github.com/FluxML/OneHotArrays.jl/issues/28 -- MethodError: getindex(::OneHotArrays.OneHotMatrix{UInt32, CuArray{UInt32, 1, CUDA.Mem.DeviceBuffer}}, ::Int64, ::Int64) is ambiguous
  end
  q2.x.storage === nothing
  q2.y.indices.storage === nothing

  @test collect(pre2) isa Vector{NamedTuple{(:x, :y)}}
  @test_broken collect(post2) isa Vector{NamedTuple{(:x, :y)}}  # collect makes no sense, but check eltype?

  @test_throws Exception gpu(((x = Flux.DataLoader(X), y = Y),))
end
