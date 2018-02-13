mutable struct Cached{F,A}
  f::F
  buffer::A
end

function (c::Cached)(args...)
  sh = shape(c.f, shape(args)...)
  bytes(sh) > length(c.buffer) && (c.buffer = similar(c.buffer, bytes(sh)))
  y = restructure(sh, c.buffer)
  inplace!(c.f, y, args...)
end
