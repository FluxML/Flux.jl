# [Tutorial: A Simple ConvNet](@id man-convnet-tutorial)

In this tutorial, we build a simple Convolutional Neural Network (ConvNet) to classify the MNIST dataset. This model has a simple architecture with three feature detection layers (Conv -> ReLU -> MaxPool) followed by a final dense layer that classifies MNIST handwritten digits. Note that this model, while simple, should hit around 99% test accuracy after training for approximately 20 epochs.
 
This example writes out the saved model to the file `mnist_conv.bson`. Also, it demonstrates basic model construction, training, saving, conditional early-exit, and learning rate scheduling.
 
To run this example, we need the following packages:
 
```julia
using Flux, MLDatasets, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy, params
using MLDatasets: MNIST
using Base.Iterators: partition
using Printf, BSON
using CUDA
CUDA.allowscalar(false)
```
 
We set default values for learning rate, batch size, number of epochs, and path for saving the file `mnist_conv.bson`:
 
```julia
Base.@kwdef mutable struct TrainArgs
   lr::Float64 = 3e-3
   epochs::Int = 20
   batch_size = 128
   savepath::String = "./"
end
```

## Data

To train our model, we need to bundle images together with their labels and group them into mini-batches (makes the training process faster). We define the function `make_minibatch` that takes as inputs the images (`X`) and their labels (`Y`) as well as the indices for the mini-batches (`idx`):
 
```julia
function make_minibatch(X, Y, idxs)
   X_batch = Array{Float32}(undef, size(X)[1:end-1]..., 1, length(idxs))
   for i in 1:length(idxs)
       X_batch[:, :, :, i] = Float32.(X[:,:,idxs[i]])
   end
   Y_batch = onehotbatch(Y[idxs], 0:9)
   return (X_batch, Y_batch)
end
```

`make_minibatch` takes the following steps:

* Creates the `X_batch` array of size `28x28x1x128` to store the mini-batches. 
* Stores the mini-batches in `X_batch`.
* One hot encodes the labels of the images.
* Stores the labels in `Y_batch`.


 
 `get_processed_data` loads the train and test data from `Flux.Data.MNIST`. First, it loads the images and labels of the train data set, and creates an array that contains the indices of the train images that correspond to each mini-batch (of size `args.batch_size`). Then, it calls the `make_minibatch` function to create all of the train mini-batches. Finally, it loads the test images and creates one mini-batch that contains them all.
 
```julia
function get_processed_data(args)
   # Load labels and images
   train_imgs, train_labels = MNIST.traindata()
   mb_idxs = partition(1:length(train_labels), args.batch_size)
   train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]
  
   # Prepare test set as one giant minibatch:
   test_imgs, test_labels = MNIST.testdata()
   test_set = make_minibatch(test_imgs, test_labels, 1:length(test_labels))
 
   return train_set, test_set
 
end
```

## Model
 
Now, we define the `build_model` function that creates a ConvNet model which is composed of *three* convolution layers (feature detection) and *one* classification layer. The input layer size is `28x28`. The images are grayscale, which means there is only *one* channel (compared to 3 for RGB) in every data point. Combined together, the convolutional layer structure would look like `Conv(kernel, input_channels => output_channels, ...)`. Each convolution layer reduces the size of the image by applying the Rectified Linear unit (ReLU) and MaxPool operations.
On the other hand, the classification layer outputs a vector of 10 dimensions (a dense layer), that is, the number of classes that the model will be able to predict.
 
 
```julia
function build_model(args; imgsize = (28,28,1), nclasses = 10)
   cnn_output_size = Int.(floor.([imgsize[1]/8,imgsize[2]/8,32])) 
 
   return Chain(
   # First convolution, operating upon a 28x28 image
   Conv((3, 3), imgsize[3]=>16, pad=(1,1), relu),
   MaxPool((2,2)),
 
   # Second convolution, operating upon a 14x14 image
   Conv((3, 3), 16=>32, pad=(1,1), relu),
   MaxPool((2,2)),
 
   # Third convolution, operating upon a 7x7 image
   Conv((3, 3), 32=>32, pad=(1,1), relu),
   MaxPool((2,2)),
 
   # Reshape 3d array into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
   flatten,
   Dense(prod(cnn_output_size), 10))
end
```

To chain the layers of a model we use the Flux function [Chain](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Chain). It enables us to call the layers in sequence on a given input. Also, we use the function [flatten](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.flatten) to reshape the output image from the last convolution layer. Finally, we call the [Dense](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense) function to create the classification layer.

## Training

Before training our model, we need to define a few functions that will be helpful for the process:

* `augment` adds gaussian random noise to our image, to make it more robust:
* `anynan` checks whether any element of the params is NaN or not:
* `accuracy` computes the proportion of inputs `x` correctly classified by our ConvNet:

```julia
augment(x) = x .+ gpu(0.1f0*randn(eltype(x), size(x)))
anynan(x) = any(y -> any(isnan, y), x)
accuracy(x, y, model) = mean(onecold(cpu(model(x))) .== onecold(cpu(y)))
```

Finally, we define the `train` function:
 
```julia
function train(; kws...)   
   args = TrainArgs(; kws...)
 
   @info("Loading data set")
   train_set, test_set = get_processed_data(args)
 
   # Define our model.  We will use a simple convolutional architecture with
   # three iterations of Conv -> ReLU -> MaxPool, followed by a final Dense layer.
   @info("Building model...")
   model = build_model(args)
 
   # Load model and datasets onto GPU, if enabled
   train_set = gpu.(train_set)
   test_set = gpu.(test_set)
   model = gpu(model)
  
   # Make sure our model is nicely precompiled before starting our training loop
   model(train_set[1][1])
 
   # `loss()` calculates the crossentropy loss between our prediction `y_hat`
   # (calculated from `model(x)`) and the ground truth `y`.  We augment the data
   # a bit, adding gaussian random noise to our image to make it more robust.
   function loss(x, y)   
       x̂ = augment(x)
       ŷ = model(x̂)
       return logitcrossentropy(ŷ, y)
   end
  
   # Train our model with the given training set using the Adam optimiser and
   # printing out performance against the test set as we go.
   opt = Adam(args.lr)
  
   @info("Beginning training loop...")
   best_acc = 0.0
   last_improvement = 0
   for epoch_idx in 1:args.epochs
       # Train for a single epoch
       Flux.train!(loss, params(model), train_set, opt)
      
       # Terminate on NaN
       if anynan(Flux.params(model))
           @error "NaN params"
           break
       end
  
       # Calculate accuracy:
       acc = accuracy(test_set..., model)
      
       @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
       # If our accuracy is good enough, quit out.
       if acc >= 0.999
           @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
           break
       end
  
       # If this is the best accuracy we've seen so far, save the model out
       if acc >= best_acc
           @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
           BSON.@save joinpath(args.savepath, "mnist_conv.bson") params=cpu.(params(model)) epoch_idx acc
           best_acc = acc
           last_improvement = epoch_idx
       end
  
       # If we haven't seen improvement in 5 epochs, drop our learning rate:
       if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
           opt.eta /= 10.0
           @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
 
           # After dropping learning rate, give it a few epochs to improve
           last_improvement = epoch_idx
       end
  
       if epoch_idx - last_improvement >= 10
           @warn(" -> We're calling this converged.")
           break
       end
   end
end
```
 
`train` calls the functions we defined above and trains our model. It stops when the model achieves 99% accuracy (early-exiting) or after performing 20 steps. More specifically, it performs the following steps:
 
   * Loads the MNIST dataset.
   * Builds our ConvNet model (as described above).
   * Loads the train and test data sets as well as our model onto a GPU (if available).
   * Defines a `loss` function that calculates the crossentropy between our prediction and the ground truth.
   * Sets the [Adam optimiser](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.Adam) to train the model with learning rate `args.lr`.
   * Runs the training loop. For each step (or epoch), it executes the following:
       * Calls `Flux.train!` function to execute one training step.
       * If any of the parameters of our model is `NaN`, then the training process is terminated.
       * Calculates the model accuracy.
       * If the model accuracy is >= 0.999, then early-exiting is executed.
       * If the actual accuracy is the best so far, then the model is saved to `mnist_conv.bson`. Also, the new best accuracy and the current epoch is saved.
       * If there has not been any improvement for the last 5 epochs, then the learning rate is dropped and the process waits a little longer for the accuracy to improve.
       * If the last improvement was more than 10 epochs ago, then the process is terminated.


## Testing

Finally, to test our model we define the `test` function: 

```julia
function test(; kws...)
   args = TrainArgs(; kws...)
  
   # Loading the test data
   _,test_set = get_processed_data(args)
  
   # Re-constructing the model with random initial weights
   model = build_model(args)
  
   # Loading the saved parameters
   BSON.@load joinpath(args.savepath, "mnist_conv.bson") params
  
   # Loading parameters onto the model
   Flux.loadparams!(model, params)
  
   test_set = gpu.(test_set)
   model = gpu(model)
   @show accuracy(test_set...,model)
end
```

`test` loads the MNIST test data set, reconstructs the model, and loads the saved parameters (in `mnist_conv.bson`) onto it. Finally, it computes our model's predictions for the test set and shows the test accuracy (around 99%).
 
To see the full version of this example, see [Simple ConvNets - model-zoo](https://github.com/FluxML/model-zoo/blob/master/vision/conv_mnist/conv_mnist.jl).
 
## Resources

* [Neural Networks in Flux.jl with Huda Nassar (working with the MNIST dataset)](https://youtu.be/Oxi0Pfmskus)
* [Convolutional Neural Networks (CNNs / ConvNets)](https://cs231n.github.io/convolutional-networks/).
* [Convolutional Neural Networks Tutorial in PyTorch](https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/).
 
!!! info
    Originally published at [fluxml.ai](https://fluxml.ai/tutorials/) on 7 February 2021.
    Written by Elliot Saba, Adarsh Kumar, Mike J Innes, Dhairya Gandhi, Sudhanshu Agrawal, Sambit Kumar Dash, fps.io, Carlo Lucibello, Andrew Dinhobl, Liliana Badillo
