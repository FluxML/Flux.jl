module CUDA

using ..CuArrays
import ..CuArrays.CUDAdrv: CuPtr, CU_NULL
using Pkg.TOML

function version_check()
  minor_version = 9
  project = joinpath(dirname(pathof(CuArrays)), "../Project.toml")
  project = TOML.parse(String(read(project)))
  version = VersionNumber(get(project, "version", "0.0.0"))
  if !(version.major == 0 && version.minor == minor_version)
    @warn """
    Flux is only supported with CuArrays v0.$minor_version.
    Try running `] pin CuArrays@0.$minor_version`.
    """
  end
end

version_check()

if !applicable(CuArray{UInt8}, undef, 1)
  (T::Type{<:CuArray})(::UndefInitializer, sz...) = T(sz...)
end

if CuArrays.libcudnn != nothing
  if isdefined(CuArrays, :libcudnn_handle)
    handle() = CuArrays.libcudnn_handle[]
  else
    handle() = CuArrays.CUDNN.handle()
  end
  include("curnn.jl")
  include("cudnn.jl")
else
  @warn("CUDNN is not installed, some functionality will not be available.")
end

end
