# render trees in Juno

@render Juno.Inline t::Tree begin
  render(t) = Juno.Tree(t.value, render.(t.children))
  Juno.Tree(typeof(t), [render(t)])
end
