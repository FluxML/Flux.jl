# Logistic Regression with MNIST

This walkthrough example will take you through writing a multi-layer perceptron that classifies MNIST digits with high accuracy.

First, we load the data using the MNIST package:

```julia
using Flux, MNIST

data = [(trainfeatures(i), onehot(trainlabel(i), 0:9)) for i = 1:60_000]
train = data[1:50_000]
test = data[50_001:60_000]
```

The only Flux-specific function here is `onehot`, which takes a class label and turns it into a one-hot-encoded vector that we can use for training. For example:

```julia
julia> onehot(:b, [:a, :b, :c])
3-element Array{Int64,1}:
 0
 1
 0
```

Otherwise, the format of the data is simple enough, it's just a list of tuples from input to output. For example:

```julia
julia> data[1]
([0.0,0.0,0.0, … 0.0,0.0,0.0],[0,0,0,0,0,1,0,0,0,0])
```

`data[1][1]` is a `28*28 == 784` length vector (mostly zeros due to the black background) and `data[1][2]` is its classification.

Now we define our model, which will simply be a function from one to the other.

```julia
m = Chain(
  Input(784),
  Affine(128), relu,
  Affine( 64), relu,
  Affine( 10), softmax)

model = tf(m)
```

We can try this out on our data already:

```julia
julia> model(data[1][1])
10-element Array{Float64,1}:
 0.10614  
 0.0850447
 0.101474
 ...
```

The model gives a probability of about 0.1 to each class – which is a way of saying, "I have no idea". This isn't too surprising as we haven't shown it any data yet. This is easy to fix:

```julia
Flux.train!(model, train, test, η = 1e-4)
```

The training step takes about 5 minutes (to make it faster we can do smarter things like batching). If you run this code in Juno, you'll see a progress meter, which you can hover over to see the remaining computation time.

Towards the end of the training process, Flux will have reported that the accuracy of the model is now about 90%. We can try it on our data again:

```julia
10-element Array{Float32,1}:
 ...
 5.11423f-7
 0.9354     
 3.1033f-5  
 0.000127077
 ...
```

Notice the class at 93%, suggesting our model is very confident about this image. We can use `onecold` to compare the true and predicted classes:

```julia
julia> onecold(data[1][2], 0:9)
5

julia> onecold(model(data[1][1]), 0:9)
5
```

Success!
