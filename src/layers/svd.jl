import LinearAlgebra: svd
import LinearAlgebra
using ..Tracker: @grad, data, track, TrackedTuple
import ..Tracker: _forward

function svd_back(U, S, V, dU, dS, dV)
    NS = length(S)
    S2 = S.^2
    Sinv = 1 ./ S
    F = S2' .- S2
    @. F = F/(F^2 + 1e-12)

    UdU = U'*dU
    VdV = V'*dV

    Su = (F.*(UdU-UdU'))*LinearAlgebra.Diagonal(S)
    Sv = LinearAlgebra.Diagonal(S) * (F.*(VdV-VdV'))

    U * (Su + Sv + LinearAlgebra.Diagonal(dS)) * V' +
    (LinearAlgebra.I - U*U') * dU*LinearAlgebra.Diagonal(Sinv) * V' +
    U*LinearAlgebra.Diagonal(Sinv) * dV' * (LinearAlgebra.I - V*V')
end

"""
    svd(A::TrackedArray) -> TrackedTuple

Return tracked tuple of (U, S, V) that `A == USV'`.
"""
svd(A::TrackedArray) = track(svd, A)
function _forward(::typeof(svd), a)
    U, S, V = svd(data(a))
    (U, S, Matrix(V)), Δ -> (svd_back(U, S, V, Δ...),)
end
