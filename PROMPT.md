In https://github.com/FluxML/Flux.jl/pull/2665 I implemented support
for gpu caching allocator in Flux. 

Supposedly this should have addressed the problems in 
https://github.com/FluxML/Flux.jl/issues/2523
and referenced issues. 

Create a perf/caching-allocator folder, and benchmark the performance of the caching allocator, checking if it really solves all memory and performance issues collected. 

