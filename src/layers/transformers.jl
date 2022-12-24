using Flux
using Flux:softmax

struct Attention{D<:Integer, K_Type, V_Type}
    d_k::D
    K::K_Type
    V::V_Type
end

Attention(d::Pair{<:Integer,<:Integer}; d_k::Integer=50) = Attention(d_k, Flux.rand(d_k,d.first), Flux.rand(d_k, d.second))
Attention(d_i::Integer, d_o::Integer; d_k::Integer=50) = Attention(d_i=>d_o; d_k)
Flux.@functor Attention (K,V,)

(m::Attention)(Q) = (softmax((Q'*m.K')/√(m.d_k))*m.V)'
