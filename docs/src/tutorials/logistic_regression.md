# Logistic Regression

The following page contains a step-by-step walkthrough of the logistic regression algorithm in Julia using Flux. We will then create a simple logistic regression model without any usage of Flux and compare the different working parts with Flux's implementation. 

Let's start by importing the required Julia packages.

```jldoctest logistic_regression
julia> using Flux, Statistics, MLDatasets, DataFrames, OneHotArrays
```

## Dataset
Let's start by importing a dataset from MLDatasets.jl. We will use the `Iris` dataset that contains the data of three different `Iris` species. The data consists of 150 data points (`x`s), each having four features. Each of these `x` is mapped to `y`, the name of a particular `Iris` specie. The following code will download the `Iris` dataset when run for the first time.

```jldoctest logistic_regression
julia> Iris()
dataset Iris:
  metadata   =>    Dict{String, Any} with 4 entries
  features   =>    150×4 DataFrame
  targets    =>    150×1 DataFrame
  dataframe  =>    150×5 DataFrame

julia> x, y = Iris(as_df=false)[:];
```

Let's have a look at our dataset -

```jldoctest logistic_regression
julia> y
1×150 Matrix{InlineStrings.String15}:
 "Iris-setosa"  "Iris-setosa"  …  "Iris-virginica"  "Iris-virginica"

julia> x |> summary
"4×150 Matrix{Float64}"
```

The `y` values here corresponds to a type of iris plant, with a total of 150 data points. The `x` values depict the sepal length, sepal width, petal length, and petal width (all in `cm`) of 150 iris plant (hence the matrix size `4×150`). Different type of iris plants have different lengths and widths of sepals and petals associated with them, and there is a definitive pattern for this in nature. We can leverage this to train a simple classifier that outputs the type of iris plant using the length and width of sepals and petals as inputs.

Our next step would be to convert this data into a form that can be fed to a machine learning model. The `x` values are arranged in a matrix and should ideally be converted to `Float32` type (see [Performance tips](@ref man-performance-tips)), but the labels must be one hot encoded. [Here](https://discourse.julialang.org/t/all-the-ways-to-do-one-hot-encoding/64807) is a great discourse thread on different techniques that can be used to one hot encode data with or without using any external Julia package.

```jldoctest logistic_regression
julia> x = Float32.(x);

julia> y = vec(y);

julia> custom_y_onehot = unique(y) .== permutedims(y)
3×150 BitMatrix:
 1  1  1  1  1  1  1  1  1  1  1  1  1  …  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0     1  1  1  1  1  1  1  1  1  1  1  1
```

This same operation can also be performed using [OneHotArrays](https://github.com/FluxML/OneHotArrays.jl)' `onehotbatch` function. We will use both of these outputs parallelly to show how intuitive FluxML is!

```jldoctest logistic_regression
julia> flux_y_onehot = onehotbatch(y, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
3×150 OneHotMatrix(::Vector{UInt32}) with eltype Bool:
 1  1  1  1  1  1  1  1  1  1  1  1  1  …  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅     ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅     1  1  1  1  1  1  1  1  1  1  1  1
```

Our data is ready. The next step would be to build a classifier for the same.

## Building a model

A logistic regression model is defined mathematically as -

```math
model(x) = σ(Wx + b)
```

where `W` is the weight matrix, `b` is the bias vector, and `σ` is any activation function. For our case, let's use the `softmax` activation function as we will be performing a multiclass classification task.

```jldoctest logistic_regression
julia> m(W, b, x) = W*x .+ b
m (generic function with 1 method)
```

Note that this model lacks an activation function, but we will come back to that.

We can now move ahead to initialize the parameters of our model. Given that our model has four inputs (4 features in every data point), and three outputs (3 different classes), the parameters can be initialized in the following way -

```jldoctest logistic_regression; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> W = rand(Float32, 3, 4);

julia> b = [0.0f0, 0.0f0, 0.0f0];
```

Now our model can take in the complete dataset and predict the class of each `x` in one go. But, we need to ensure that our model outputs the probabilities of an input belonging to the respective classes. As our model has three outputs, each would denote the probability of the input belonging to a particular class.

We will use an activation function to map our outputs to a probability value. It would make sense to use a `softmax` activation function here, which is defined mathematically as -

```math
σ(\vec{x}) = \frac{\\e^{z_{i}}}{\\sum_{j=1}^{k} \\e^{z_{j}}}
```

The `softmax` function scales down the outputs to probability values such that the sum of all the final outputs equals `1`. Let's implement this in Julia.

```jldoctest logistic_regression
julia> custom_softmax(x) = exp.(x) ./ sum(exp.(x), dims=1)
custom_softmax (generic function with 1 method)
```

The implementation looks straightforward enough! Note that we specify `dims=1` in the `sum` function to calculate the sum of probabilities in each column. Remember, we will have a 3×150 matrix (predicted `y`s) as the output of our model, where each column would be an output of a corresponding input.

Let's combine this `softmax` function with our model to construct the complete `custom_model`.

```jldoctest logistic_regression
julia> custom_model(W, b, x) = m(W, b, x) |> custom_softmax
custom_model (generic function with 1 method)
```

Let's check if our model works.

```jldoctest logistic_regression
julia> custom_model(W, b, x) |> size
(3, 150)
```

It works! Let's check if the `softmax` function is working.

```jldoctest logistic_regression
julia> all(0 .<= custom_model(W, b, x) .<= 1)
true

julia> sum(custom_model(W, b, x), dims=1)
1×150 Matrix{Float32}:
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  …  1.0  1.0  1.0  1.0  1.0  1.0  1.0
```

Every output value is between `0` and `1`, and every column adds to `1`!

Let's convert our `custom_model` to a Flux model. Flux provides the users with a very elegant API that almost feels like writing your code!

Note, all the `flux_*` variables in this tutorial would be general, that is, they can be used as it is with some other similar-looking dataset, but the `custom_*` variables will remain specific to this tutorial.

```jldoctest logistic_regression
julia> flux_model = Chain(Dense(4 => 3), softmax)
Chain(
  Dense(4 => 3),                        # 15 parameters
  NNlib.softmax,
)
```

A [`Dense(4 => 3)`](@ref Dense) layer denotes a layer with four inputs (four features in every data point) and three outputs (three classes or labels). This layer is the same as the mathematical model defined by us above. Under the hood, Flux too calculates the output using the same expression, but we don't have to initialize the parameters ourselves this time, instead Flux does it for us.

The `softmax` function provided by NNLib.jl is re-exported by Flux, which has been used here. Lastly, Flux provides users with a `Chain` struct which makes stacking layers seamless.

A model's weights and biases can be accessed as follows -

```jldoctest logistic_regression; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> flux_model[1].weight, flux_model[1].bias
(Float32[0.78588694 -0.45968163 -0.77409476 0.2358028; -0.9049773 -0.58643705 0.466441 -0.79523873; 0.82426906 0.4143493 0.7630932 0.020588955], Float32[0.0, 0.0, 0.0])
```

We can now pass the complete data in one go, with each data point having four features (four inputs)!

## Loss and accuracy

Our next step should be to define some quantitative values for our model, which we will maximize or minimize during the complete training procedure. These values will be the loss function and the accuracy metric.

Let's start by defining a loss function, a `logitcrossentropy` function.

```jldoctest logistic_regression
julia> custom_logitcrossentropy(ŷ, y) = mean(.-sum(y .* logsoftmax(ŷ; dims = 1); dims = 1));
```

Now we can wrap the `custom_logitcrossentropy` inside a function that takes in the model parameters, `x`s, and `y`s, and returns the loss value.

```jldoctest logistic_regression; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> function custom_loss(W, b, x, y)
           ŷ = custom_model(W, b, x)
           custom_logitcrossentropy(ŷ, y)
       end;

julia> custom_loss(W, b, x, custom_y_onehot)
1.1714406827505623
```

The loss function works!

Flux provides us with many minimal yet elegant loss functions. In fact, the `custom_logitcrossentropy` defined above has been taken directly from Flux. The functions present in Flux includes sanity checks, ensures efficient performance, and behaves well with the overall FluxML ecosystem.

```jldoctest logistic_regression; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> function flux_loss(flux_model, x, y)
           ŷ = flux_model(x)
           Flux.logitcrossentropy(ŷ, y)
       end;

julia> flux_loss(flux_model, x, flux_y_onehot)
1.2156688659673647
```

Next, let's define an accuracy function, which we will try to maximize during our training procedure. Before jumping to accuracy, let's define a `onecold` function. The `onecold` function would convert our output, which remember, are probability values, to the actual class names.

We can divide this task into two parts -
1. Identify the index of the maximum element of each column in the output matrix
2. Convert this index to a class name

The maximum index should be calculated along the columns (remember, each column is the output of a single `x` data point). We can use Julia's `findmax` function to achieve this.

```jldoctest logistic_regression; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> findmax(custom_y_onehot, dims=1)
(Bool[1 1 … 1 1], CartesianIndex{2}[CartesianIndex(1, 1) CartesianIndex(1, 2) … CartesianIndex(3, 149) CartesianIndex(3, 150)])

julia> mxidx = findmax(custom_y_onehot, dims=1)[2]
1×150 Matrix{CartesianIndex{2}}:
 CartesianIndex(1, 1)  CartesianIndex(1, 2)  …  CartesianIndex(3, 150)

julia> mxidx[1].I
(1, 1)

julia> mxidx[1].I[1]
1
```

Now we can write a function that iterates over our output, calculates the indices of the maximum element in each column, and maps them to a class name.  

```jldoctest logistic_regression
julia> function custom_onecold(custom_y_onehot)
           mxidx = findmax(custom_y_onehot, dims=1)[2]
           custom_y_cold = Vector{String}(undef, size(custom_y_onehot)[2])
           for i = 1:size(custom_y_onehot)[2]
               if mxidx[i].I[1] == 1
                   custom_y_cold[i] = "Iris-setosa"
               elseif mxidx[i].I[1] == 2
                   custom_y_cold[i] = "Iris-versicolor"
               elseif mxidx[i].I[1] == 3
                   custom_y_cold[i] = "Iris-virginica"
               end
           end
           custom_y_cold
       end;

julia> custom_onecold(custom_y_onehot)
150-element Vector{String}:
 "Iris-setosa"
 "Iris-setosa"
 "Iris-setosa"
 "Iris-setosa"
 "Iris-setosa"
 "Iris-setosa"
 "Iris-setosa"
 "Iris-setosa"
 "Iris-setosa"
 "Iris-setosa"
 ⋮
 "Iris-virginica"
 "Iris-virginica"
 "Iris-virginica"
 "Iris-virginica"
 "Iris-virginica"
 "Iris-virginica"
 "Iris-virginica"
 "Iris-virginica"
 "Iris-virginica"
```

It works!

Flux provides users with the `onecold` function so that we don't have to write it on our own. Let's see how our `custom_onecold` function compares to `Flux.onecold`.

```jldoctest logistic_regression
julia> istrue = Flux.onecold(flux_y_onehot, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]) .== custom_onecold(custom_y_onehot);

julia> all(istrue)
true
```

Both the functions act identically!

We now move to the `accuracy` metric and run it with the untrained `custom_model`.

```jldoctest logistic_regression; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> custom_accuracy(W, b, x, y) = mean(custom_onecold(custom_model(W, b, x)) .== y);

julia> custom_accuracy(W, b, x, y)
0.3333333333333333
```

We could also have used Flux's built-in functionality to define this accuracy function.

```jldoctest logistic_regression; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> flux_accuracy(x, y) = mean(Flux.onecold(flux_model(x), ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]) .== y);

julia> flux_accuracy(x, y)
0.24
```

## Training the model

Let's train our model using the classic Gradient Descent algorithm. According to the gradient descent algorithm, the weights and biases should be iteratively updated using the following mathematical equations -

```math
\begin{aligned}
W &= W - \eta * \frac{dL}{dW} \\
b &= b - \eta * \frac{dL}{db}
\end{aligned}
```

Here, `W` is the weight matrix, `b` is the bias vector, ``\eta`` is the learning rate, ``\frac{dL}{dW}`` is the derivative of the loss function with respect to the weight, and ``\frac{dL}{db}`` is the derivative of the loss function with respect to the bias.

The derivatives are calculated using an Automatic Differentiation tool, and Flux uses [`Zygote.jl`](https://github.com/FluxML/Zygote.jl) for the same. Since Zygote.jl is an independent Julia package, it can be used outside of Flux as well! Refer to the documentation of Zygote.jl for more information on the same.

Our first step would be to obtain the gradient of the loss function with respect to the weights and the biases. Flux re-exports Zygote's `gradient` function; hence, we don't need to import Zygote explicitly to use the functionality. `gradient` takes in a function and its arguments, and returns a tuple containing `∂f/∂x` for each argument x. Let's pass in `custom_loss` and the arguments required by `custom_loss` to `gradient`. We will require the derivatives of the loss function (`custom_loss`) with respect to the weights (`∂f/∂w`) and the bias (`∂f/∂b`) to carry out gradient descent, but we can ignore the partial derivatives of the loss function (`custom_loss`) with respect to `x` (`∂f/∂x`) and one hot encoded `y` (`∂f/∂y`).

```jldoctest logistic_regression
julia> dLdW, dLdb, _, _ = gradient(custom_loss, W, b, x, custom_y_onehot);
```

We can now update the parameters, following the gradient descent algorithm -

```jldoctest logistic_regression; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> W .= W .- 0.1 .* dLdW;

julia> b .= b .- 0.1 .* dLdb;
```

The parameters have been updated! We can now check the value of our custom loss function -

```jldoctest logistic_regression; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> custom_loss(W, b, x, custom_y_onehot)
1.164742997664842
```

The loss went down! Let's plug our super training logic inside a function.

```jldoctest logistic_regression
julia> function train_custom_model()
           dLdW, dLdb, _, _ = gradient(custom_loss, W, b, x, custom_y_onehot)
           W .= W .- 0.1 .* dLdW
           b .= b .- 0.1 .* dLdb
       end;
```

We can plug the training function inside a loop and train the model for more epochs. The loop can be tailored to suit the user's needs, and the conditions can be specified in plain Julia. Here we will train the model for a maximum of `500` epochs, but to ensure that the model does not overfit, we will break as soon as our accuracy value crosses or becomes equal to `0.98`.

```jldoctest logistic_regression; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> for i = 1:500
            train_custom_model();
            custom_accuracy(W, b, x, y) >= 0.98 && break
       end
    
julia> @show custom_accuracy(W, b, x, y);
custom_accuracy(W, b, x, y) = 0.98
```

Everything works! Our model achieved an accuracy of `0.98`! Let's have a look at the loss.

```jldoctest logistic_regression; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> custom_loss(W, b, x, custom_y_onehot)
0.6520349798243569
```

As expected, the loss went down too! Now, let's repeat the same steps with our `flux_model`.

We can write a similar-looking training loop for our `flux_model` and train it similarly.

```jldoctest logistic_regression; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> flux_loss(flux_model, x, flux_y_onehot)
1.215731131385928

julia> function train_flux_model()
           dLdm, _, _ = gradient(flux_loss, flux_model, x, flux_y_onehot)
           @. flux_model[1].weight = flux_model[1].weight - 0.1 * dLdm[:layers][1][:weight]
           @. flux_model[1].bias = flux_model[1].bias - 0.1 * dLdm[:layers][1][:bias]
       end;

julia> for i = 1:500
            train_flux_model();
            flux_accuracy(x, y) >= 0.98 && break
       end
```

Looking at the accuracy and loss value -

```jldoctest logistic_regression; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> @show flux_accuracy(x, y);
flux_accuracy(x, y) = 0.98

julia> flux_loss(flux_model, x, flux_y_onehot)
0.6952386604624324
```

We see a very similar final loss and accuracy.

---

Summarising this tutorial, we saw how we can run a logistic regression algorithm in Julia with and without using Flux. We started by importing the classic `Iris` dataset, and one hot encoded the labels. Next, we defined our model, the loss function, and the accuracy, all by ourselves.

Finally, we trained the model by manually writing down the Gradient Descent algorithm and optimising the loss. Interestingly, we implemented most of the functions on our own, and then parallelly compared them with the functionalities provided by Flux!

!!! info
    Originally published on TODO,
    by Saransh Chopra.
