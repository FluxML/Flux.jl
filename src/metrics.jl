using OneHotArrays: OneHotMatrix, onecold, onehotbatch
using Statistics: mean

"""
    multiclass_accuracy(model, test_x, test_y::Vector{T})

Used to calculate accuracy of multi-class problem.
Test set consists of `test_x` and `test_y`. `size(test_x)[end]` and `size(test_y)[end]`
equal to the number of samples. `test_x` contains batch of samples, and `test_y` is a Vector,
which contains the position of the largest element of the output (`findmax(output)`)
obtained by feeding `test_x` into the ideal model.

e.g.

```jldoctest
julia> typeof(test_x)
Matrix{Float32} (alias for Array{Float32, 2})

julia> typeof(test_y)
Vector{Int64} (alias for Array{Int64, 1})

julia> multiclass_accuracy(model, test_x, test_y);
```
"""
multiclass_accuracy(model, test_x, test_y::Vector{T}) where {T} = mean(onecold(model(test_x)) .== test_y)

"""
    multiclass_accuracy(model, test_x, test_y::Union{Matrix{T}, OneHotMatrix})

Used to calculate accuracy of multi-class problem.
Test set consists of `test_x` and `test_y`. `size(test_x)[end]` and `size(test_y)[end]`
equal to the number of samples. `test_x` contains batch of samples, and `test_y` is a Matrix
or OneHotMatrix, which contains the direct output obtained by feeding `test_x` into
the ideal model. Length of `test_y`'s elements equals to the number of classes.

e.g.

```jldoctest
julia> typeof(test_x)
Matrix{N0f8} (alias for Array{FixedPointNumbers.Normed{UInt8, 8}, 2})

julia> typeof(test_y)
OneHotMatrix{UInt32, Vector{UInt32}} (alias for OneHotArray{UInt32, 1, 2, Array{UInt32, 1}})

julia> multiclass_accuracy(model(test_x), test_y);
```
"""
multiclass_accuracy(model, test_x, test_y::Union{Matrix{T}, OneHotMatrix}) where {T} = mean(onecold(model(test_x)) .== onecold(test_y))

"""
    multilabel_accuracy(model, test_x, test_y; threshold=0.5)

Used to calculated accuracy of multi-label problem.
Test set consists of `test_x` and `test_y`. `size(test_x)[end]` and `size(test_y)[end]`
equal to the number of samples. `threshold` can be set to change the classes identified,
0.5 by default.
"""
multilabel_accuracy(model, test_x, test_y; threshold=0.5) = mean((model(test_x) .> threshold) .== test_y)
