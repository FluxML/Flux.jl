# Internal interface

struct _TrackedComplex{T<:Real}
  data::Complex{T}
  tracker::Tracked{Complex{T}}
end

_TrackedComplex(x::Complex) = _TrackedComplex(x, Tracked{typeof(x)}(Call(), zero(x)))

data(x::_TrackedComplex) = x.data
tracker(x::_TrackedComplex) = x.tracker

Base.real(x::_TrackedComplex) = track(real, x)
Base.imag(x::_TrackedComplex) = track(imag, x)

@grad real(x::_TrackedComplex) = real(data(x)), r̄ -> (r̄ + zero(r̄)*im,)
@grad imag(x::_TrackedComplex) = imag(data(x)), ī -> (zero(ī) + ī*im,)

unwrap(x::_TrackedComplex) = real(x) + imag(x)*im

track(f::Call, x::Complex) =
  unwrap(_TrackedComplex(x, Tracked{typeof(x)}(f, zero(x))))

param(x::Complex) = _TrackedComplex(float(x))

# External interface

TrackedComplex{T<:Real} = Complex{TrackedReal{T}}

data(x::TrackedComplex) = data(real(x)) + data(imag(x))*im

tracker(x::TrackedComplex) =
  Tracked{typeof(data(x))}(Call(c -> (real(c), imag(c)),
                                (tracker(real(x)),tracker(imag(x)))),
                           zero(data(x)))

function Base.show(io::IO, x::TrackedComplex)
  show(io, data(x))
  print(io, " (tracked)")
end

Base.log(x::TrackedComplex) = track(log, x)

@grad log(x::TrackedComplex) = log(data(x)), ȳ -> (ȳ/x,)
