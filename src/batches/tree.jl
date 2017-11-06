using AbstractTrees

struct Tree{T}
  value::T
  children::Vector{Tree{T}}
end

Tree{T}(x::T, xs::Tree{T}...) where T = Tree{T}(x, [xs...])
Tree{T}(x) where T = Tree(convert(T, x))

Tree(x::T, xs::Tree{T}...) where T = Tree{T}(x, xs...)

AbstractTrees.children(t::Tree) = t.children
AbstractTrees.printnode(io::IO, t::Tree) = show(io, t.value)

function Base.show(io::IO, t::Tree)
  println(io, typeof(t))
  print_tree(io, t)
end

using Juno

@render Juno.Inline t::Tree begin
  render(t) = Juno.Tree(t.value, render.(t.children))
  Juno.Tree(typeof(t), [render(t)])
end
