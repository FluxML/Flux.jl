"""
1. Title: Boston Housing Data
2. Sources:
   (a) Origin:  This dataset was taken from the StatLib library which is
                maintained at Carnegie Mellon University.
   (b) Creator:  Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the
                 demand for clean air', J. Environ. Economics & Management,
                 vol.5, 81-102, 1978.
   (c) Date: July 7, 1993
3. Number of Instances: 506
4. Number of Attributes: 13 continuous attributes (including "class"
                            attribute "MEDV"), 1 binary-valued attribute.
5. Attribute Information:
       1. CRIM      per capita crime rate by town
       2. ZN        proportion of residential land zoned for lots over
                    25,000 sq.ft.
       3. INDUS     proportion of non-retail business acres per town
       4. CHAS      Charles River dummy variable (= 1 if tract bounds
                    river; 0 otherwise)
       5. NOX       nitric oxides concentration (parts per 10 million)
       6. RM        average number of rooms per dwelling
       7. AGE       proportion of owner-occupied units built prior to 1940
       8. DIS       weighted distances to five Boston employment centres
       9. RAD       index of accessibility to radial highways
       10. TAX      full-value property-tax rate per 10,000 dollars
       11. PTRATIO  pupil-teacher ratio by town
       12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks
                    by town
       13. LSTAT    % lower status of the population
       14. MEDV     Median value of owner-occupied homes in 1000's of dollars
       Downloaded From: https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
"""
module Housing

using DelimitedFiles
using ..Data: deps, download_and_verify, deprecation_message

#Uncomment if package exists
#const cache_prefix = "https://cache.julialang.org/"
const cache_prefix = ""

function load()
    isfile(deps("housing.data")) && return

    @info "Downloading the Boston housing Dataset"
    download_and_verify("$(cache_prefix)http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                        deps("housing.data"),
                        "baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a")

    #@info "Download complete. Working on the files"
    path = deps()
    isfile(deps("housing.data")) && touch(joinpath(path, "tempfile.data"))
    open(joinpath(path, "tempfile.data"), "a") do fout
        open(deps("housing.data"), "r") do fin
            for line in eachline(fin)
                line = replace(lstrip(line), r" +" => s",")
                println(fout, line)
            end
        end
    end
    mv(joinpath(path, "tempfile.data"), deps("housing.data"), force=true)
end

"""
Gets the targets for the Boston housing dataset, a 506 element array listing the targets for each example
```julia
julia> using Flux
julia> target = Flux.Data.Housing.targets()
julia> summary(target)
506×1 Array{Float64,2}
julia> target[1]
24.0
"""
function targets()
    deprecation_message()
    load()
    housing = readdlm(deps("housing.data"), ',')
    reshape(Vector{Float64}(housing[1:end,end]), (506, 1))
end


"""
Gets the names of the features provided in the dataset
"""
function feature_names()
    ["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat"]
end


"""
Gets the features of the Boston Housing Dataset. This is a 506x13 Matrix of Float64 datatypes.
The values are in the order ["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat"].
It has 506 examples.
```julia
julia> using Flux
julia> features = Flux.Data.Housing.features()
julia> summary(features)
506×13 Array{Float64,2}
julia> features[1, :]
13-element Array{Float64,1}:
0.00632
18.0
2.31
0.0
0.538
   ⋮
296.0
15.3
396.9
4.98
"""
function features()
    deprecation_message()
    load()
    housing = readdlm(deps("housing.data"), ',')
    Matrix{Float64}(housing[1:end, 1:13])
end


end