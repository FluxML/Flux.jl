export Conv2D

struct Conv2D <: Model
  filter::Param{Array{Float64,4}} # [height, width, inchans, outchans]
  stride::Dims{2}
end

Conv2D(size; in = 1, out = 1, stride = (1,1), init = initn) =
  Conv2D(param(initn(size..., in, out)), stride)

infer(c::Conv2D, in::Dims{4}) =
  (in[1], map(i -> (in[i+1]-size(c.filter,i))÷c.stride[i]+1, (1,2))..., size(c.filter, 4))

# TODO: many of these should just be functions

for Pool in :[MaxPool, AvgPool].args
  @eval begin
    struct $Pool <: Model
      size::Dims{2}
      stride::Dims{2}
    end

    $Pool(size; stride = (1,1)) =
      $Pool(size, stride)

    infer(c::$Pool, in::Dims{4}) =
      (in[1], map(i -> (in[i+1]-c.size[i])÷c.stride[i]+1, (1,2))..., in[4])

    shape(c::$Pool) = nothing
  end
end

struct Reshape{N}
  dims::Dims{N}
end

Reshape(dims::Integer...) = Reshape(dims)

shape(r::Reshape, ::Void) = r.dims
