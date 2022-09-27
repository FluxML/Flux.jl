# Flux Neural Networks in One Minute

If you have used neural networks before, then this simple example might be helpful for seeing how the major parts of Flux work together. Try pasting the code into the REPL prompt.

If you haven't, then you might prefer the [Fitting a Straight Line](models/overview.jl) page.

```julia
# With Julia 1.7+, this will prompt if neccessary to install everything, including CUDA:
using Flux, Plots, Statistics

# Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
noisy = rand(Float32, 2, 1000)                                    # 2×1000 Matrix{Float32}
truth = map(col -> xor(col...), eachcol(noisy .> 0.5))            # 1000-element Vector{Bool}

p_true = Plots.scatter(noisy[1,:], noisy[2,:], zcolor=truth, lab="", title="True classification")

# Define our model, a multi-layer perceptron with one hidden layer of size 3:
model = Chain(Dense(2 => 3, tanh), BatchNorm(3), Dense(3 => 2), softmax)

# The model encapsulates parameters, randomly initialised. Its initial output is:
out = model(noisy)                                                # 2×1000 Matrix{Float32}
p_raw = Plots.scatter(noisy[1,:], noisy[2,:], zcolor=out[1,:], legend=false, title="Untrained network")

# To train the model, we use batches of 64 samples:
mat = Flux.onehotbatch(truth, [true, false])                      # 2×1000 OneHotMatrix
data = Flux.DataLoader((noisy, mat), batchsize=64, shuffle=true);
first(data) .|> summary                                           # ("2×64 Matrix{Float32}", "2×64 Matrix{Bool}")

pars = Flux.params(model)  # contains references to arrays in model
opt = Flux.Adam(0.01)      # will store optimiser momentum etc.

# Training loop, using whole data set 1000 times:
for epoch in 1:1_000
    Flux.train!(pars, data, opt) do x, y
        # First argument of train! is a loss function, here defined by a `do` block.
        # This gets x and y, each a 2×64 Matrix, from data, and compares:
        Flux.crossentropy(model(x), y)
    end
end

pars  # has changed!
opt
out2 = model(noisy)

mean((out2[1,:] .> 0.5) .== truth)  # accuracy 94% so far!

p_done = Plots.scatter(noisy[1,:], noisy[2,:], zcolor=out2[1,:], legend=false, title="Trained network")
plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330))  # combined plot, shown below
```

![](../assets/oneminute.png)


This XOR ("exclusive or") problem is a variant of the famous one which drove Minsky and Papert to invent deep neural networks in 1969. For small values of "deep" -- this has one hidden layer, while earlier perceptrons had none. (What they call a hidden layer, Flux calls the output of the first layer, `model[1](noisy)`.)

Since then things have developed a little. 

## Features of Note

Some things to notice in this example are:

* The batch dimension of data is always the last one. Thus a `2×1000 Matrix` is a thousand observations, each a column of length 2.

* The `model` can be called like a function, `y = model(x)`. It encapsulates the parameters (and state).

* But the model does not contain the loss function, nor the optimisation rule. Instead `Adam()` stores between iterations the momenta it needs.

* The function `train!` likes data as an iterator generating Tuples. You can use lower-level functions instead.
