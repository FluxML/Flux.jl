# [Flux Overview: Fitting a Straight Line](@id man-overview)

Flux is a pure Julia ML stack that allows you to build predictive models. Here are the steps for a typical Flux program:

1. Provide training and test data
2. Build a model with configurable *parameters* to make predictions
3. Iteratively train the model by tweaking the parameters to improve predictions
4. Verify your model

Under the hood, Flux uses a technique called automatic differentiation to take gradients that help improve predictions. Flux is also fully written in Julia so you can easily replace any layer of Flux with your own code to improve your understanding or satisfy special requirements.

Here's how you'd use Flux to build and train the most basic of models, step by step.

### A Trivial Prediction

This example will predict the output of the function `4x + 2`. Making such predictions is called "linear regression", and is really too simple to *need* a neural network. But it's a nice toy example.

First, import `Flux` and define the function we want to simulate:

```jldoctest overview
julia> using Flux

julia> actual(x) = 4x + 2
actual (generic function with 1 method)
```

This example will build a model to approximate the `actual` function.

## 1. Provide Training and Test Data

Use the `actual` function to build sets of data for training and verification:

```jldoctest overview
julia> x_train, x_test = hcat(0:5...), hcat(6:10...)
([0 1 … 4 5], [6 7 … 9 10])

julia> y_train, y_test = actual.(x_train), actual.(x_test)
([2 6 … 18 22], [26 30 … 38 42])
```

Normally, your training and test data come from real world observations, but here we simulate them.

## 2. Build a Model to Make Predictions

Now, build a model to make predictions with `1` input and `1` output:

```jldoctest overview; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> model = Dense(1 => 1)
Dense(1 => 1)       # 2 parameters

julia> model.weight
1×1 Matrix{Float32}:
 0.95041317

julia> model.bias
1-element Vector{Float32}:
 0.0
```

Under the hood, a dense layer is a struct with fields `weight` and `bias`. `weight` represents a weights' matrix and `bias` represents a bias vector. There's another way to think about a model. In Flux, *models are conceptually predictive functions*: 

```jldoctest overview
julia> predict = Dense(1 => 1)
Dense(1 => 1)       # 2 parameters
```

`Dense(1 => 1)` also implements the function `σ(Wx+b)` where `W` and `b` are the weights and biases. `σ` is an activation function (more on activations later). Our model has one weight and one bias, but typical models will have many more. Think of weights and biases as knobs and levers Flux can use to tune predictions. Activation functions are transformations that tailor models to your needs. 

This model will already make predictions, though not accurate ones yet:

```jldoctest overview; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> predict(x_train)
1×6 Matrix{Float32}:
 0.0  0.906654  1.81331  2.71996  3.62662  4.53327
```

In order to make better predictions, you'll need to provide a *loss function* to tell Flux how to objectively *evaluate* the quality of a prediction. Loss functions compute the cumulative distance between actual values and predictions. 

```jldoctest overview; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> using Statistics

julia> loss(model, x, y) = mean(abs2.(model(x) .- y));

julia> loss(predict, x_train, y_train)
122.64734f0
```

More accurate predictions will yield a lower loss. You can write your own loss functions or rely on those already provided by Flux. This loss function is called [mean squared error](https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/mean-squared-error/) (and built-in as [`mse`](@ref Flux.Losses.mse)). Flux works by iteratively reducing the loss through *training*.

## 3. Improve the Prediction

Under the hood, the Flux [`Flux.train!`](@ref) function uses *a loss function* and *training data* to improve the *parameters* of your model based on a pluggable [`optimiser`](../training/optimisers.md):

```jldoctest overview
julia> using Flux: train!

julia> opt = Descent()
Descent(0.1)

julia> data = [(x_train, y_train)]
1-element Vector{Tuple{Matrix{Int64}, Matrix{Int64}}}:
 ([0 1 … 4 5], [2 6 … 18 22])
```

Now, we have the optimiser and data we'll pass to `train!`. All that remains are the parameters of the model. Remember, each model is a Julia struct with a function and configurable parameters. Remember, the dense layer has weights and biases that depend on the dimensions of the inputs and outputs: 

```jldoctest overview; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> predict.weight
1×1 Matrix{Float32}:
 0.9066542

julia> predict.bias
1-element Vector{Float32}:
 0.0
```

The dimensions of these model parameters depend on the number of inputs and outputs.

Flux will adjust predictions by iteratively changing these parameters according to the optimizer.

This optimiser implements the classic gradient descent strategy. Now improve the parameters of the model with a call to [`Flux.train!`](@ref) like this:

```jldoctest overview
julia> train!(loss, predict, data, opt)
```

And check the loss:

```jldoctest overview; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> loss(predict, x_train, y_train)
116.38745f0
```

It went down. Why? 

```jldoctest overview; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> predict.weight, predict.bias
(Float32[7.5777884], Float32[1.9466728])
```

The parameters have changed. This single step is the essence of machine learning.

## 3+. Iteratively Train the Model

In the previous section, we made a single call to `train!` which iterates over the data we passed in just once. An *epoch* refers to one pass over the dataset. Typically, we will run the training for multiple epochs to drive the loss down even further. Let's run it a few more times:

```jldoctest overview; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> for epoch in 1:200
         train!(loss, predict, data, opt)
       end

julia> loss(predict, x_train, y_train)
0.00339581f0

julia> predict.weight, predict.bias
(Float32[4.0178537], Float32[2.0050256])
```

After 200 training steps, the loss went down, and the parameters are getting close to those in the function the model is built to predict.

## 4. Verify the Results

Now, let's verify the predictions:

```jldoctest overview; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> predict(x_test)
1×5 Matrix{Float32}:
 26.1121  30.13  34.1479  38.1657  42.1836

julia> y_test
1×5 Matrix{Int64}:
 26  30  34  38  42
```

The predictions are good. Here's how we got there. 

First, we gathered real-world data into the variables `x_train`, `y_train`, `x_test`, and `y_test`. The `x_*` data defines inputs, and the `y_*` data defines outputs. The `*_train` data is for training the model, and the `*_test` data is for verifying the model. Our data was based on the function `4x + 2`.

Then, we built a single input, single output predictive model, `predict = Dense(1 => 1)`. The initial predictions weren't accurate, because we had not trained the model yet.

After building the model, we trained it with `train!(loss, predict, data, opt)`. The loss function is first, followed by the model itself, the training data, and the `Descent` optimizer provided by Flux. We ran the training step once, and observed that the parameters changed and the loss went down. Then, we ran the `train!` many times to finish the training process.

After we trained the model, we verified it with the test data to verify the results. 

This overall flow represents how Flux works. Let's drill down a bit to understand what's going on inside the individual layers of Flux.
