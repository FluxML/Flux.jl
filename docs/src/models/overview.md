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

```julia>
julia> using Flux

julia> actual(x) = 4x + 2
actual (generic function with 1 method)
```

This example will build a model to approximate the `actual` function.

## Provide Training and Test Data

Use the `actual` function to build sets of data for training and verification:

```julia>
julia> x_train, x_test = hcat(0:5...), hcat(6:10...)
([0 1 … 4 5], [6 7 … 9 10])

julia> y_train, y_test = actual.(x_train), actual.(y_train)
([2 6 … 18 22], [26 30 … 38 42])
```

Normally, your training and test data come from real world observations, but this function will simulate real-world observations.

## Build a Model to Make Predictions

Now, build a model to make predictions with `1` input and `1` output:

```julia>
julia> predict = Dense(1, 1)
Dense(1, 1)
```

A dense layer implements the function `σ(Wx+b)`, where `W` represents a weight, `b` represents a bias, and `σ` is an activation function (more on activations later). Our model has one weight and one bias, but typical models will have many more. Think of weights and biases as knobs and levers Flux can use to tune predictions. Activation functions are transformations that tailor models to your needs.

This model will already make predictions, though not accurate ones yet:

```julia>
julia> predict(x_train)
1×6 Array{Float32,2}:
 0.0  -0.990091  -1.98018  -2.97027  -3.96036  -4.95045
```

In order to make better predictions, you'll need to provide a *loss function* to tell Flux how to objectively *evaluate* the quality of a prediction. Loss functions compute the cumulative distance between actual values and predictions. 

```julia>
julia> loss(x, y) = Flux.Losses.mse(predict(x), y)
loss (generic function with 1 method)

julia> loss(x_train, y_train)
282.1601f0
```

More accurate predictions will yield a lower loss. You can write your own loss functions or rely on those already provided by Flux. This loss function is called [mean squared error](https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/mean-squared-error/). Flux works by iteratively reducing the loss through *training*.

## Improve the Prediction

Under the hood, the Flux [`train!`](../training/training.md) function uses *a loss function* and *training data* to improve the *parameters* of your model based on a pluggable [`optimiser`](../training/optimisers.md):

```julia>
julia> using Flux: train!

julia> opt = Descent()
Descent(0.1)

julia> data = [(x_train, y_train)]
1-element Array{Tuple{Array{Int64,2},Array{Int64,2}},1}:
 ([0 1 … 4 5], [2 6 … 18 22])
```

Now, we have the optimiser and data we'll pass to `train!`. All that remains are the parameters of the model. Each model is a Julia struct with a function and configurable parameters. Remember, the dense layer has weights and biases that depend on the dimensions of the inputs and outputs: 

```julia>
julia> predict.W
1-element Array{Float64,1}:
 -0.99009055

julia> predict.b
1-element Array{Float64,1}:
 0.0
```

The dimensions of these model parameters depend on the number of inputs and outputs. Since models can have hundreds of inputs and several layers, it helps to have a function to collect the parameters into the data structure Flux expects:

```julia>
julia> parameters = params(predict)
Params([Float32[-0.99009055], Float32[0.0]])
```

These are the parameters Flux will change, one step at a time, to improve predictions. The first parameter is the weight and the second is the bias. Flux will shape predictions by iteratively changing these parameters.

This optimiser implements the iconic gradient descent strategy. Now improve the parameters of the model with a call to [`train!`](https://fluxml.ai/Flux.jl/stable/training/training/#Flux.Optimise.train!) like this:

```julia>
julia> train!(loss, parameters, data, opt)
```

And check the loss:

```julia>
julia> loss(x_train, y_train)
267.8037f0
```

It went down. 

## Iteratively Train the Model

Let's run it a few more times:

```julia>
julia> for epoch in 1:200
         train!(loss, parameters, data, opt)
       end

julia> loss(x_train, y_train)
0.0518891f0
```

After 200 training steps, the loss went down. 

## Verify the Results

Now, let's verify the predictions:

```julia>
julia> loss(x_test, y_test)
0.0518891f0

julia> predict(y_test)
1×5 Array{Float32,2}:
 106.713  122.821  138.929  155.038  171.146

julia> actual.(y_test)
1×5 Array{Int64,2}:
 106  122  138  154  170
```

The predictions are good. Let's drill down a bit to understand what's going on inside the individual layers of Flux.
