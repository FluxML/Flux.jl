struct OneHotVector <: AbstractVector{Bool}
  ix::UInt32
  of::UInt32
end

Base.size(xs::OneHotVector) = (Int64(xs.of),)

Base.getindex(xs::OneHotVector, i::Integer) = i == xs.ix

Base.:*(A::AbstractMatrix, b::OneHotVector) = A[:, b.ix]

struct OneHotMatrix{A<:AbstractVector{OneHotVector}} <: AbstractMatrix{Bool}
  data::A
end

Base.size(xs::OneHotMatrix) = (Int64(length(xs.data[1])),length(xs.data))

Base.getindex(xs::OneHotMatrix, i::Int, j::Int) = xs.data[j][i]

Base.:*(A::AbstractMatrix, B::OneHotMatrix) = A[:, map(x->x.ix, B.data)]

Base.hcat(x::OneHotVector, xs::OneHotVector...) = OneHotMatrix([x, xs...])

@require CuArrays begin
  import CuArrays: CuArray, cudaconvert
  CuArrays.cu(xs::OneHotMatrix) = OneHotMatrix(CuArrays.cu(xs.data))
  Base.Broadcast._containertype(::Type{<:OneHotMatrix{<:CuArray}}) = CuArray
  cudaconvert(x::OneHotMatrix{<:CuArray}) = OneHotMatrix(cudaconvert(x.data))
end

@require CLArrays begin
    import CLArrays.Shorthands: cl
    using CLArrays: CLArray, GlobalArray, GlobalPointer, PreDeviceArray
    cl(xs::OneHotMatrix) = OneHotMatrix(cl(xs.data))
    # the on device conversions are still a bit complicated...
    CLArrays.kernel_convert(x::OneHotMatrix{T}) where T <: CLArray = OneHotMatrix(CLArrays.kernel_convert(x.data))
    CLArrays.predevice_type(::Type{OneHotMatrix{T}}) where T <: GlobalArray = OneHotMatrix{CLArrays.predevice_type(T)}
    CLArrays.device_type(x::OneHotMatrix{T}) where T <: CLArray = OneHotMatrix{CLArrays.device_type(x.data)}
    CLArrays.reconstruct(x::OneHotMatrix{T}, ptr::GlobalPointer) where T <: PreDeviceArray = OneHotMatrix(CLArrays.reconstruct(x.data, ptr))

    CLArrays.GPUArrays.arg_length(x::OneHotMatrix{T}) where T <: CLArrays.GPUArrays.GPUArray = UInt32.(size(x))
end

onehot(l, labels) = OneHotVector(findfirst(labels, l), length(labels))
onehotbatch(ls, labels) = OneHotMatrix([onehot(l, labels) for l in ls])

argmax(y::AbstractVector, labels = 1:length(y)) =
  labels[findfirst(y, maximum(y))]

argmax(y::AbstractMatrix, l...) =
  squeeze(mapslices(y -> argmax(y, l...), y, 1), 1)
