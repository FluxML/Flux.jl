using AbstractTrees

struct Tree{T}
  value::T
  children::Vector{Tree{T}}
end

Tree(x::T, xs::Vector{Tree{T}} = Tree{T}[]) where T = Tree{T}(x, xs)
Tree(x::T, xs::Tree{T}...) where T = Tree{T}(x, [xs...])

AbstractTrees.children(t::Tree) = t.children
AbstractTrees.printnode(io::IO, t::Tree) = show(io, t.value)
