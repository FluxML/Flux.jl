# DataLoader

Flux provides the `DataLoader` type in the `Flux.Data` module to handle iteration over mini-batches of data. 
Any dataset type implementing the `LearnBase.nobs` and `LearnBase.getobs` interfaces can be handled
by the DataLoader. Flux define his own `nobs`/`getobs` specialized methods handling common types
such as arrays, tuples, named tuples, and dictionaries.

```@docs
Flux.Data.DataLoader
LearnBase.nobs
LearnBase.getobs
```
