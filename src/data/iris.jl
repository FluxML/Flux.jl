
"""

    Iris

Fisher's classic iris dataset.

Measurements from 3 different species of iris: setosa, versicolor and 
virginica.  There are 50 examples of each species.

There are 4 measurements for each example: sepal length, sepal width, petal 
length and petal width.  The measurements are in centimeters.

The module retrieves the data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

"""
module Iris

using DelimitedFiles
using ..Data: deps, download_and_verify

const cache_prefix = ""

# Uncomment if the iris.data file is cached to cache.julialang.org.
# const cache_prefix = "https://cache.julialang.org/"

function load()
    isfile(deps("iris.data")) && return

    @info "Downloading iris dataset."
    download_and_verify("$(cache_prefix)https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                        deps("iris.data"),
                        "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0")
end

"""

    labels()

Get the labels of the iris dataset, a 150 element array of strings listing the 
species of each example.

```jldoctest
julia> labels = Flux.Data.Iris.labels();

julia> summary(labels)
"150-element Array{String,1}"

julia> labels[1]
"Iris-setosa"
```
"""
function labels()
    load()
    iris = readdlm(deps("iris.data"), ',')
    Vector{String}(iris[1:end, end])
end

"""

    features()

Get the features of the iris dataset.  This is a 4x150 matrix of Float64 
elements.  It has a row for each feature (sepal length, sepal width, 
petal length, petal width) and a column for each example.

```jldoctest
julia> features = Flux.Data.Iris.features();

julia> summary(features)
"4×150 Array{Float64,2}"

julia> features[:, 1]
4-element Array{Float64,1}:
 5.1
 3.5
 1.4
 0.2
```
"""
function features()
    load()
    iris = readdlm(deps("iris.data"), ',')
    Matrix{Float64}(iris[1:end, 1:4]')
end
end


