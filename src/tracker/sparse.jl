function multrans(A::Matrix{Tv}, B::SparseMatrixCSC{Tv}) where Tv
  mA, nA = size(A)
  mB, nB = size(B)
  @boundscheck nA == nB || throw(DimensionMismatch())
  C = zeros(Tv, mA, mB)
  @inbounds for jB in 1:nB
      for kB in B.colptr[jB]:(B.colptr[jB+1] - 1)
          iB = B.rowval[kB]
          xB = B.nzval[kB]
          for iA in 1:mA
              C[iA, iB] += A[iA, jB] * xB
          end
      end
  end
  return C
end

function mul(A::Matrix{Tv}, B::SparseMatrixCSC{Tv}) where Tv
  mA, nA = size(A)
  mB, nB = size(B)
  @boundscheck nA == mB || throw(DimensionMismatch())
  C = zeros(Tv, mA, nB)
  @inbounds for jB in 1:nB
      for kB in B.colptr[jB]:(B.colptr[jB+1] - 1)
          iB = B.rowval[kB]
          xB = B.nzval[kB]
          for iA in 1:mA
              C[iA, jB] += A[iA, iB] * xB
          end
      end
  end
  return C
end

back(::typeof(*), Δ, a::AbstractMatrix, b::SparseMatrixCSC) = @back(a, multrans(Δ,b))
back(::typeof(mul), Δ, a::AbstractMatrix, b::SparseMatrixCSC) = @back(a, multrans(Δ,b))
a::TrackedMatrix * b::SparseMatrixCSC = TrackedArray(Call(mul, a, b))

