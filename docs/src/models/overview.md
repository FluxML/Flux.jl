# Flux Overview

Flux is a pure Julia ML stack that allows you to build predictive models. Here are the steps for a typical Flux program:

- Provide training and test data
- Build a model with configurable *parameters* to make predictions
- Iteratively train the model on training data by tweaking the parameters to improve predictions
- Verify your model on test data

Under the hood, Flux uses a technique called automatic differentiation to take gradients that help improve predictions. Flux is also fully written in Julia so you can easily replace any layer of Flux with your own code to improve your understanding or satisfy special requirements.

Here's how you'd use Flux to build and train the most basic of models, step by step.

This example will predict the output of the simple function `4x + 2`. First, import `Flux` and define the function we want to learn from training data:

```julia
julia> using Flux

julia> actual(x) = 4x  + 2
actual (generic function with 1 method)
```

This example will build a model to approximate the `actual` function.

## Provide Training and Test Data

Use the `actual` function to build sets of data for training and verification:

```julia
julia> x_train, x_test = [0 1 2 3 4 5], [6 7 8 9 10]
([0 1 … 4 5], [6 7 … 9 10])

julia> y_train, y_test = actual.(x_train), actual.(x_test)
([2 6 … 18 22], [26 30 … 38 42])
```

Normally, your training and test data come from real world observations, but this function will simulate real-world observations.

## Build a Model to Make Predictions

Now, build a model to make predictions with `1` input and `1` output:

```julia
julia> model = Dense(1, 1)
Dense(1, 1)

julia> model.weight
1×1 Matrix{Float32}:
 -0.067352235

julia> model.bias
1-element Vector{Float32}:
 0.0
```

Under the hood, a `Dense` layer is a struct with fields `weight` and `bias`. `weight` is a matrix of model weights and `bias` represents a bias vector. By default, these model parameters are initialized randomly.

`Dense(1, 1)` also implements the function `σ(Wx+b)` where `W` and `b` are the weights and biases. `σ` is an activation function that makes the mapping from inputs to output nonlinear. Our model has one weight and one bias, but typical models will have many more. Think of weights and biases as knobs and levers Flux can use to tune predictions.

This model will already make predictions, though not accurate ones since we have not trained the model yet:

```julia
julia> model(x_train)
1×6 Matrix{Float32}:
 0.0  -0.0673522  -0.134704  -0.202057  -0.269409  -0.336761
```

## Learn Model Parameters by Optimisation

In order to make better predictions, you'll need to provide a *loss function* to tell Flux how to objectively *evaluate* the quality of a prediction. Loss functions compute the average distance between actual values and predictions. 

```julia
julia> loss(x, y) = Flux.Losses.mse(model(x), y)
loss (generic function with 1 method)

julia> loss(x_train, y_train)
196.32093811035156
```

More accurate predictions will yield a lower loss. This loss function is called [mean squared error](https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/mean-squared-error/). You can write your own loss functions or rely on those already provided by Flux. Flux works by iteratively reducing the loss through *training*.

The Flux [`train!`](@ref) function uses *a loss function* and *training data* to improve the *parameters* of your model based on an [`optimiser`](../training/optimisers.md). We use the gradient `Descent` optimiser:

```julia
julia> using Flux: train!

julia> opt = Descent()
Descent(0.1)

julia> data = [(x_train, y_train)]
1-element Array{Tuple{Array{Int64,2},Array{Int64,2}},1}:
 ([0 1 … 4 5], [2 6 … 18 22])
```

We now have the loss function, optimiser and data that can be pass on to `train!`. All that remains to complete the specification are the parameters of the model, which `train!` expects to be a particular data structure that Flux sets up using the `params` function:

```
julia> parameters = params(model)
Params([[-0.067352235], [0.0]])
```

The [`train!`](@ref) function in Flux will use the optimizer to iteratively adjust the weights and biases in `parameters` to minimize the loss function. Calling `train!` once makes the optimizer take a single step toward the minimum: 

```
julia> train!(loss, parameters, data, opt)
```

We can check that the loss has decreased:

```
julia> loss(x_train, y_train)
186.32362365722656
```

as the parameters were changed by the algorithm:

```
julia> parameters
Params([[8.389461], [2.4336762]])
```

In the previous section, we made a single call to `train!` which iterates over the data we passed in just once. An *epoch* refers to one pass over the dataset. Typically, we will run the training for multiple epochs to drive the loss down even further. Let's run it a few more times:

```
julia> for epoch in 1:200
         train!(loss, parameters, data, opt)
       end

julia> loss(x_train, y_train)
0.0054461159743368626

julia> parameters
Params([[4.022609], [2.0063674]])
```

After 200 training steps, the loss went down, and the parameters are getting close to those in the function the model is built to predict.

## Evaluate the Predictive Performance of the Model

Now, let's verify the predictions:

```
julia> model(x_test)
1×5 Array{Float64,2}:
 26.142  30.1646  34.1872  38.2099  42.2325

julia> y_test
1×5 Array{Int64,2}:
 26  30  34  38  42
```

As expected, the predictions are very accurate since the underlying function was simple and observed without noise. Later on we will see `Flux` in action on substantially more challenging applications.

In summary, Flux builds and evaluates a predictive model through the following steps:

1. Setup training and test data.
2. Build a predictive model with initialized parameters.
3. Find better parameters by optimizing a loss function that penalizes poor predictions.
4. Evaluate the learned model by assessing predictive performance on the test data.

Let's drill down a bit to understand what's going on inside the individual layers of Flux.
