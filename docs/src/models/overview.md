# Flux Overview

Flux is a pure Julia ML stack that allows you to build predictive models from data. Flux is fully written in Julia so you can easily replace any layer of Flux with your own code to improve your understanding or satisfy special requirements.

A typical Flux program consists of four steps:

1. **Set up data** for training and test 
2. **Build a model** with configurable parameters to make predictions
3. **Learn the model parameters** on training data by iteratively updating the parameters to improve predictions
4. **Evaluate the model** on test data

Under the hood, Flux uses a technique called automatic differentiation to compute the gradients needed in Step 3 to update the parameters for improved predictions. 

In this short tutorial we will walk you through the four steps of a Flux program to build and train a model to predict the output of the simple function `4x + 2`:

```julia
julia> actual(x) = 4x + 2
actual (generic function with 1 method)
```

## Step 1 - Set up data for training and test

Normally, your training and test data come from real world observations, but we will here use the `actual` function to build sets of data for training and model validation:

```julia
julia> x_train, x_test = [0 1 2 3 4 5], [6 7 8 9 10];

julia> y_train, y_test = actual.(x_train), actual.(x_test)
([2 6 … 18 22], [26 30 … 38 42])
```

## Step 2 - Build a model with configurable parameters

Now, use `Flux` to build a model to make predictions with `1` input and `1` output:

```julia
julia> using Flux

julia> model = Dense(1, 1)
Dense(1, 1)

julia> model.weight
1×1 Matrix{Float32}:
 -1.4925033

julia> model.bias
1-element Vector{Float32}:
 0.0
```

Under the hood, a `Dense` layer is a struct with fields `weight` and `bias`. The field `weight` holds a matrix of model weights and `bias` represents a bias vector. By default, these model parameters are initialized randomly, and will later be learned from data by optimization. 

`Dense(1, 1)` also implements the function `σ(Wx+b)` where `W` and `b` are the weights and biases. `σ` is an activation function that applies a (possibly nonlinear) transformation. Our model has one weight and one bias, but typical models will have many more. Think of weights and biases as knobs and levers Flux can use to tune predictions.

Models are conceptually predictive functions in Flux, i.e. we can call our `model` function on input data to make predictions. We can try out this already now, but the predictions will not be accurate since we have not trained the model yet:

```julia
julia> model(x_train)
1×6 Matrix{Float32}:
 0.0  -1.4925  -2.98501  -4.47751  -5.97001  -7.46252
```

## Step 3 - Learn the model parameters on training data 

In order to make better predictions, you'll need to provide a *loss function* to tell Flux how to objectively *evaluate* the quality of a prediction. Generally, loss functions compute the average distance between actual values and predictions so that more accurate predictions will yield a lower loss. Here is an example with the [mean squared error](https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/mean-squared-error/) loss function:

```julia
julia> loss(x, y) = Flux.Losses.mse(model(x), y)
loss (generic function with 1 method)

julia> loss(x_train, y_train)
282.16010605766024
```

You can write your own loss functions or rely on those already provided by Flux. 

The Flux [`train!`](@ref) function uses *a loss function* and *training data* to improve the *parameters* of your model based on an [`optimiser`](../training/optimisers.md). We use the gradient [`Descent`](@ref) optimiser:

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
Params([[-0.99009055], [0.0]])
```

These are the parameters Flux will change, one step at a time, to improve predictions. Each of the parameters comes from `model`: 

```
julia> model.weight in parameters, model.bias in parameters
(true, true)
```

The [`train!`](@ref) function in Flux will use the optimizer to iteratively adjust the weights and biases in `parameters` to minimize the loss function. Calling `train!` once makes the optimizer take a single step toward the minimum of our loss function:

```
julia> train!(loss, parameters, data, opt)
```

This call to `train!` updated the parameters to:

```
julia> parameters
Params([[9.158408791666668], [2.895045275]])
```

and the loss has indeed decreased as a result:

```
julia> loss(x_train, y_train)
267.8037f0
```



A single call to `train!` iterates over the data we passed in just once. An *epoch* refers to one pass over the dataset. Typically, we will run the training for multiple epochs to drive the loss down even further. Let's run it a few more times:

```
julia> for epoch in 1:200
         train!(loss, parameters, data, opt)
       end

julia> loss(x_train, y_train)
0.007433314787010791

julia> parameters
Params([[3.9735880692372345], [1.9925541368157165]])
```

After 200 training steps, the loss went down, and the parameters are getting close to those in the function the model is built to predict.

## Step 4 - Evaluate the model on test data

Now, let's evaluate the predictions by comparing them to the true outputs in `y_test`:

```
julia> model(x_test)
1×5 Array{Float64,2}:
 25.8442  29.8194  33.7946  37.7698  41.745

julia> y_test
1×5 Array{Int64,2}:
 26  30  34  38  42
```

As expected, the predictions are very accurate since the underlying function was simple and observed without noise. Later on we will see Flux in action on substantially more challenging applications.

This was a short demo of how Flux can be easily used to:
- set up a machine learning model using the `Dense` function in Flux
- learn the model parameters from training data, `x_train` and `y_train`, using gradient `Descent` to iteratively minimize the mean squared error loss function `Flux.Losses.mse`
- make predictions by calling the model on test input data `x_test`.
- evaluate the model by comparing its predictions to the actual test outputs `y_test`

Let's now drill down a bit to understand what's going on inside the individual layers of Flux.
