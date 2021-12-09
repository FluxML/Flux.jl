# Flux Overview

Flux is a pure Julia ML stack that allows you to build predictive models. Here are the steps for a typical Flux program:

- Provide training and test data
- Build a model with configurable *parameters* to make predictions
- Iteratively train the model by tweaking the parameters to improve predictions
- Verify your model

Under the hood, Flux uses a technique called automatic differentiation to take gradients that help improve predictions. Flux is also fully written in Julia so you can easily replace any layer of Flux with your own code to improve your understanding or satisfy special requirements.

Here's how you'd use Flux to build and train the most basic of models, step by step.

## Make a Trivial Prediction

This example will predict the output of the function `4x + 2`. First, import `Flux` and define the function we want to simulate:

```julia
julia> using Flux

julia> actual(x) = 4x + 2
actual (generic function with 1 method)
```

This example will build a model to approximate the `actual` function.

## Provide Training and Test Data

Use the `actual` function to build sets of data for training and verification:

```julia
julia> x_train, x_test = hcat(0:5...), hcat(6:10...)
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
 -1.4925033

julia> model.bias
1-element Vector{Float32}:
 0.0
```

Under the hood, a dense layer is a struct with fields `weight` and `bias`. `weight` represents a weights' matrix and `bias` represents a bias vector. There's another way to think about a model. In Flux, *models are conceptually predictive functions*: 

```julia
julia> predict = Dense(1, 1)
```

`Dense(1, 1)` also implements the function `σ(Wx+b)` where `W` and `b` are the weights and biases. `σ` is an activation function (more on activations later). Our model has one weight and one bias, but typical models will have many more. Think of weights and biases as knobs and levers Flux can use to tune predictions. Activation functions are transformations that tailor models to your needs. 

This model will already make predictions, though not accurate ones yet:

```julia
julia> predict(x_train)
1×6 Matrix{Float32}:
 0.0  -1.4925  -2.98501  -4.47751  -5.97001  -7.46252
```

In order to make better predictions, you'll need to provide a *loss function* to tell Flux how to objectively *evaluate* the quality of a prediction. Loss functions compute the cumulative distance between actual values and predictions. 

```julia
julia> loss(x, y) = Flux.Losses.mse(predict(x), y)
loss (generic function with 1 method)

julia> loss(x_train, y_train)
282.16010605766024
```

More accurate predictions will yield a lower loss. You can write your own loss functions or rely on those already provided by Flux. This loss function is called [mean squared error](https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/mean-squared-error/). Flux works by iteratively reducing the loss through *training*.

## Improve the Prediction

Under the hood, the Flux [`Flux.train!`](@ref) function uses *a loss function* and *training data* to improve the *parameters* of your model based on a pluggable [`optimiser`](../training/optimisers.md):

```julia
julia> using Flux: train!

julia> opt = Descent()
Descent(0.1)

julia> data = [(x_train, y_train)]
1-element Array{Tuple{Array{Int64,2},Array{Int64,2}},1}:
 ([0 1 … 4 5], [2 6 … 18 22])
```

Now, we have the optimiser and data we'll pass to `train!`. All that remains are the parameters of the model. Remember, each model is a Julia struct with a function and configurable parameters. Remember, the dense layer has weights and biases that depend on the dimensions of the inputs and outputs: 

```julia
julia> predict.weight
1-element Array{Float64,1}:
 -0.99009055

julia> predict.bias
1-element Array{Float64,1}:
 0.0
```

The dimensions of these model parameters depend on the number of inputs and outputs. Since models can have hundreds of inputs and several layers, it helps to have a function to collect the parameters into the data structure Flux expects:

```
julia> parameters = Flux.params(predict)
Params([[-0.99009055], [0.0]])
```

These are the parameters Flux will change, one step at a time, to improve predictions. Each of the parameters comes from the `predict` model: 

```
julia> predict.weight in parameters, predict.bias in parameters
(true, true)

```

The first parameter is the weight and the second is the bias. Flux will adjust predictions by iteratively changing these parameters according to the optimizer.

This optimiser implements the classic gradient descent strategy. Now improve the parameters of the model with a call to [`Flux.train!`](@ref) like this:

```
julia> train!(loss, parameters, data, opt)
```

And check the loss:

```
julia> loss(x_train, y_train)
267.8037f0
```

It went down. Why? 

```
julia> parameters
Params([[9.158408791666668], [2.895045275]])
```

The parameters have changed. This single step is the essence of machine learning.

## Iteratively Train the Model

In the previous section, we made a single call to `train!` which iterates over the data we passed in just once. An *epoch* refers to one pass over the dataset. Typically, we will run the training for multiple epochs to drive the loss down even further. Let's run it a few more times:

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

## Verify the Results

Now, let's verify the predictions:

```
julia> predict(x_test)
1×5 Array{Float64,2}:
 25.8442  29.8194  33.7946  37.7698  41.745

julia> y_test
1×5 Array{Int64,2}:
 26  30  34  38  42
```

The predictions are good. Here's how we got there. 

First, we gathered real-world data into the variables `x_train`, `y_train`, `x_test`, and `y_test`. The `x_*` data defines inputs, and the `y_*` data defines outputs. The `*_train` data is for training the model, and the `*_test` data is for verifying the model. Our data was based on the function `4x + 2`.

Then, we built a single input, single output predictive model, `predict = Dense(1, 1)`. The initial predictions weren't accurate, because we had not trained the model yet.

After building the model, we trained it with `train!(loss, parameters, data, opt)`. The loss function is first, followed by the `parameters` holding the weights and biases of the model, the training data, and the `Descent` optimizer provided by Flux. We ran the training step once, and observed that the parameters changed and the loss went down. Then, we ran the `train!` many times to finish the training process.

After we trained the model, we verified it with the test data to verify the results. 

This overall flow represents how Flux works. Let's drill down a bit to understand what's going on inside the individual layers of Flux.
