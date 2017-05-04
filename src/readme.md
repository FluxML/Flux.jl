`model.jl` implements the core Flux API for forward and backward passes.

The `compiler` folder implements the `@net` macro and some dataflow manipulation (like loop unrolling). The `mxnet` and `tensorflow` folders in `backend` each describe how to run `@net` code on those backends. These are the most involved parts of Flux where the magic-y stuff happens, but everything else is pretty straightforward Julia code.

`layers` is Flux's "standard library" of model parts; layers, activations, cost functions. Most are implemented using `@net`. `control.jl` implements `Chain` and others in pure Julia. `shims.jl` includes some things like convolutions which don't have "real" implementations yet, but can compile to a backend regardless.

`dims` implements the mechanisms for typing batch (and other) dimensions. This is basically standalone.

`data.jl` and `utils.jl` implement misc utilities.
