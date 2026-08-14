var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Flux:-The-Julia-Machine-Learning-Library-1",
    "page": "Home",
    "title": "Flux: The Julia Machine Learning Library",
    "category": "section",
    "text": "Flux is a library for machine learning. It comes \"batteries-included\" with many useful tools built in, but also lets you use the full power of the Julia language where you need it. The whole stack is implemented in clean Julia code (right down to the GPU kernels) and any part can be tweaked to your liking."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "Install Julia 0.6.0 or later, if you haven\'t already.Pkg.add(\"Flux\")\n# Optional but recommended\nPkg.update() # Keep your packages up to date\nPkg.test(\"Flux\") # Check things installed correctlyStart with the basics. The model zoo is also a good starting point for many common kinds of models."
},

{
    "location": "models/basics.html#",
    "page": "Basics",
    "title": "Basics",
    "category": "page",
    "text": ""
},

{
    "location": "models/basics.html#Model-Building-Basics-1",
    "page": "Basics",
    "title": "Model-Building Basics",
    "category": "section",
    "text": ""
},

{
    "location": "models/basics.html#Taking-Gradients-1",
    "page": "Basics",
    "title": "Taking Gradients",
    "category": "section",
    "text": "Consider a simple linear regression, which tries to predict an output array y from an input x. (It\'s a good idea to follow this example in the Julia repl.)W = rand(2, 5)\nb = rand(2)\n\npredict(x) = W*x .+ b\nloss(x, y) = sum((predict(x) .- y).^2)\n\nx, y = rand(5), rand(2) # Dummy data\nloss(x, y) # ~ 3To improve the prediction we can take the gradients of W and b with respect to the loss function and perform gradient descent. We could calculate gradients by hand, but Flux will do it for us if we tell it that W and b are trainable parameters.using Flux.Tracker\n\nW = param(W)\nb = param(b)\n\nl = loss(x, y)\n\nback!(l)loss(x, y) returns the same number, but it\'s now a tracked value that records gradients as it goes along. Calling back! then calculates the gradient of W and b. We can see what this gradient is, and modify W to train the model.W.grad\n\n# Update the parameter\nW.data .-= 0.1(W.grad)\n\nloss(x, y) # ~ 2.5The loss has decreased a little, meaning that our prediction x is closer to the target y. If we have some data we can already try training the model.All deep learning in Flux, however complex, is a simple generalisation of this example. Of course, models can look very different – they might have millions of parameters or complex control flow, and there are ways to manage this complexity. Let\'s see what that looks like."
},

{
    "location": "models/basics.html#Building-Layers-1",
    "page": "Basics",
    "title": "Building Layers",
    "category": "section",
    "text": "It\'s common to create more complex models than the linear regression above. For example, we might want to have two linear layers with a nonlinearity like sigmoid (σ) in between them. In the above style we could write this as:W1 = param(rand(3, 5))\nb1 = param(rand(3))\nlayer1(x) = W1 * x .+ b1\n\nW2 = param(rand(2, 3))\nb2 = param(rand(2))\nlayer2(x) = W2 * x .+ b2\n\nmodel(x) = layer2(σ.(layer1(x)))\n\nmodel(rand(5)) # => 2-element vectorThis works but is fairly unwieldy, with a lot of repetition – especially as we add more layers. One way to factor this out is to create a function that returns linear layers.function linear(in, out)\n  W = param(randn(out, in))\n  b = param(randn(out))\n  x -> W * x .+ b\nend\n\nlinear1 = linear(5, 3) # we can access linear1.W etc\nlinear2 = linear(3, 2)\n\nmodel(x) = linear2(σ.(linear1(x)))\n\nmodel(x) # => 2-element vectorAnother (equivalent) way is to create a struct that explicitly represents the affine layer.struct Affine\n  W\n  b\nend\n\nAffine(in::Integer, out::Integer) =\n  Affine(param(randn(out, in)), param(randn(out)))\n\n# Overload call, so the object can be used as a function\n(m::Affine)(x) = m.W * x .+ m.b\n\na = Affine(10, 5)\n\na(rand(10)) # => 5-element vectorCongratulations! You just built the Dense layer that comes with Flux. Flux has many interesting layers available, but they\'re all things you could have built yourself very easily.(There is one small difference with Dense – for convenience it also takes an activation function, like Dense(10, 5, σ).)"
},

{
    "location": "models/basics.html#Stacking-It-Up-1",
    "page": "Basics",
    "title": "Stacking It Up",
    "category": "section",
    "text": "It\'s pretty common to write models that look something like:layer1 = Dense(10, 5, σ)\n# ...\nmodel(x) = layer3(layer2(layer1(x)))For long chains, it might be a bit more intuitive to have a list of layers, like this:using Flux\n\nlayers = [Dense(10, 5, σ), Dense(5, 2), softmax]\n\nmodel(x) = foldl((x, m) -> m(x), x, layers)\n\nmodel(rand(10)) # => 2-element vectorHandily, this is also provided for in Flux:model2 = Chain(\n  Dense(10, 5, σ),\n  Dense(5, 2),\n  softmax)\n\nmodel2(rand(10)) # => 2-element vectorThis quickly starts to look like a high-level deep learning library; yet you can see how it falls out of simple abstractions, and we lose none of the power of Julia code.A nice property of this approach is that because \"models\" are just functions (possibly with trainable parameters), you can also see this as simple function composition.m = Dense(5, 2) ∘ Dense(10, 5, σ)\n\nm(rand(10))Likewise, Chain will happily work with any Julia function.m = Chain(x -> x^2, x -> x+1)\n\nm(5) # => 26"
},

{
    "location": "models/basics.html#Layer-helpers-1",
    "page": "Basics",
    "title": "Layer helpers",
    "category": "section",
    "text": "Flux provides a set of helpers for custom layers, which you can enable by callingFlux.treelike(Affine)This enables a useful extra set of functionality for our Affine layer, such as collecting its parameters or moving it to the GPU."
},

{
    "location": "models/recurrence.html#",
    "page": "Recurrence",
    "title": "Recurrence",
    "category": "page",
    "text": ""
},

{
    "location": "models/recurrence.html#Recurrent-Models-1",
    "page": "Recurrence",
    "title": "Recurrent Models",
    "category": "section",
    "text": ""
},

{
    "location": "models/recurrence.html#Recurrent-Cells-1",
    "page": "Recurrence",
    "title": "Recurrent Cells",
    "category": "section",
    "text": "In the simple feedforward case, our model m is a simple function from various inputs xᵢ to predictions yᵢ. (For example, each x might be an MNIST digit and each y a digit label.) Each prediction is completely independent of any others, and using the same x will always produce the same y.y₁ = f(x₁)\ny₂ = f(x₂)\ny₃ = f(x₃)\n# ...Recurrent networks introduce a hidden state that gets carried over each time we run the model. The model now takes the old h as an input, and produces a new h as output, each time we run it.h = # ... initial state ...\nh, y₁ = f(h, x₁)\nh, y₂ = f(h, x₂)\nh, y₃ = f(h, x₃)\n# ...Information stored in h is preserved for the next prediction, allowing it to function as a kind of memory. This also means that the prediction made for a given x depends on all the inputs previously fed into the model.(This might be important if, for example, each x represents one word of a sentence; the model\'s interpretation of the word \"bank\" should change if the previous input was \"river\" rather than \"investment\".)Flux\'s RNN support closely follows this mathematical perspective. The most basic RNN is as close as possible to a standard Dense layer, and the output is also the hidden state.Wxh = randn(5, 10)\nWhh = randn(5, 5)\nb   = randn(5)\n\nfunction rnn(h, x)\n  h = tanh.(Wxh * x .+ Whh * h .+ b)\n  return h, h\nend\n\nx = rand(10) # dummy data\nh = rand(5)  # initial hidden state\n\nh, y = rnn(h, x)If you run the last line a few times, you\'ll notice the output y changing slightly even though the input x is the same.We sometimes refer to functions like rnn above, which explicitly manage state, as recurrent cells. There are various recurrent cells available, which are documented in the layer reference. The hand-written example above can be replaced with:using Flux\n\nrnn2 = Flux.RNNCell(10, 5)\n\nx = rand(10) # dummy data\nh = rand(5)  # initial hidden state\n\nh, y = rnn2(h, x)"
},

{
    "location": "models/recurrence.html#Stateful-Models-1",
    "page": "Recurrence",
    "title": "Stateful Models",
    "category": "section",
    "text": "For the most part, we don\'t want to manage hidden states ourselves, but to treat our models as being stateful. Flux provides the Recur wrapper to do this.x = rand(10)\nh = rand(5)\n\nm = Flux.Recur(rnn, h)\n\ny = m(x)The Recur wrapper stores the state between runs in the m.state field.If you use the RNN(10, 5) constructor – as opposed to RNNCell – you\'ll see that it\'s simply a wrapped cell.julia> RNN(10, 5)\nRecur(RNNCell(Dense(15, 5)))"
},

{
    "location": "models/recurrence.html#Sequences-1",
    "page": "Recurrence",
    "title": "Sequences",
    "category": "section",
    "text": "Often we want to work with sequences of inputs, rather than individual xs.seq = [rand(10) for i = 1:10]With Recur, applying our model to each element of a sequence is trivial:m.(seq) # returns a list of 5-element vectorsThis works even when we\'ve chain recurrent layers into a larger model.m = Chain(LSTM(10, 15), Dense(15, 5))\nm.(seq)"
},

{
    "location": "models/recurrence.html#Truncating-Gradients-1",
    "page": "Recurrence",
    "title": "Truncating Gradients",
    "category": "section",
    "text": "By default, calculating the gradients in a recurrent layer involves the entire history. For example, if we call the model on 100 inputs, calling back! will calculate the gradient for those 100 calls. If we then calculate another 10 inputs we have to calculate 110 gradients – this accumulates and quickly becomes expensive.To avoid this we can truncate the gradient calculation, forgetting the history.truncate!(m)Calling truncate! wipes the slate clean, so we can call the model with more inputs without building up an expensive gradient computation.truncate! makes sense when you are working with multiple chunks of a large sequence, but we may also want to work with a set of independent sequences. In this case the hidden state should be completely reset to its original value, throwing away any accumulated information. reset! does this for you."
},

{
    "location": "models/regularisation.html#",
    "page": "Regularisation",
    "title": "Regularisation",
    "category": "page",
    "text": ""
},

{
    "location": "models/regularisation.html#Regularisation-1",
    "page": "Regularisation",
    "title": "Regularisation",
    "category": "section",
    "text": "Applying regularisation to model parameters is straightforward. We just need to apply an appropriate regulariser, such as vecnorm, to each model parameter and add the result to the overall loss.For example, say we have a simple regression.m = Dense(10, 5)\nloss(x, y) = crossentropy(softmax(m(x)), y)We can regularise this by taking the (L2) norm of the parameters, m.W and m.b.penalty() = vecnorm(m.W) + vecnorm(m.b)\nloss(x, y) = crossentropy(softmax(m(x)), y) + penalty()When working with layers, Flux provides the params function to grab all parameters at once. We can easily penalise everything with sum(vecnorm, params).julia> params(m)\n2-element Array{Any,1}:\n param([0.355408 0.533092; … 0.430459 0.171498])\n param([0.0, 0.0, 0.0, 0.0, 0.0])\n\njulia> sum(vecnorm, params(m))\n26.01749952921026 (tracked)Here\'s a larger example with a multi-layer perceptron.m = Chain(\n  Dense(28^2, 128, relu),\n  Dense(128, 32, relu),\n  Dense(32, 10), softmax)\n\nloss(x, y) = crossentropy(m(x), y) + sum(vecnorm, params(m))\n\nloss(rand(28^2), rand(10))"
},

{
    "location": "models/layers.html#",
    "page": "Model Reference",
    "title": "Model Reference",
    "category": "page",
    "text": ""
},

{
    "location": "models/layers.html#Basic-Layers-1",
    "page": "Model Reference",
    "title": "Basic Layers",
    "category": "section",
    "text": "These core layers form the foundation of almost all neural networks.Chain\nDense\nConv2D"
},

{
    "location": "models/layers.html#Flux.RNN",
    "page": "Model Reference",
    "title": "Flux.RNN",
    "category": "function",
    "text": "RNN(in::Integer, out::Integer, σ = tanh)\n\nThe most basic recurrent layer; essentially acts as a Dense layer, but with the output fed back into the input each time step.\n\n\n\n"
},

{
    "location": "models/layers.html#Flux.LSTM",
    "page": "Model Reference",
    "title": "Flux.LSTM",
    "category": "function",
    "text": "LSTM(in::Integer, out::Integer, σ = tanh)\n\nLong Short Term Memory recurrent layer. Behaves like an RNN but generally exhibits a longer memory span over sequences.\n\nSee this article for a good overview of the internals.\n\n\n\n"
},

{
    "location": "models/layers.html#Flux.Recur",
    "page": "Model Reference",
    "title": "Flux.Recur",
    "category": "type",
    "text": "Recur(cell)\n\nRecur takes a recurrent cell and makes it stateful, managing the hidden state in the background. cell should be a model of the form:\n\nh, y = cell(h, x...)\n\nFor example, here\'s a recurrent network that keeps a running total of its inputs.\n\naccum(h, x) = (h+x, x)\nrnn = Flux.Recur(accum, 0)\nrnn(2) # 2\nrnn(3) # 3\nrnn.state # 5\nrnn.(1:10) # apply to a sequence\nrnn.state # 60\n\n\n\n"
},

{
    "location": "models/layers.html#Recurrent-Layers-1",
    "page": "Model Reference",
    "title": "Recurrent Layers",
    "category": "section",
    "text": "Much like the core layers above, but can be used to process sequence data (as well as other kinds of structured data).RNN\nLSTM\nFlux.Recur"
},

{
    "location": "models/layers.html#NNlib.σ",
    "page": "Model Reference",
    "title": "NNlib.σ",
    "category": "function",
    "text": "σ(x) = 1 / (1 + exp(-x))\n\nClassic sigmoid activation function.\n\n\n\n"
},

{
    "location": "models/layers.html#NNlib.relu",
    "page": "Model Reference",
    "title": "NNlib.relu",
    "category": "function",
    "text": "relu(x) = max(0, x)\n\nRectified Linear Unit activation function.\n\n\n\n"
},

{
    "location": "models/layers.html#NNlib.leakyrelu",
    "page": "Model Reference",
    "title": "NNlib.leakyrelu",
    "category": "function",
    "text": "leakyrelu(x) = max(0.01x, x)\n\nLeaky Rectified Linear Unit activation function. You can also specify the coefficient explicitly, e.g. leakyrelu(x, 0.01).\n\n\n\n"
},

{
    "location": "models/layers.html#NNlib.elu",
    "page": "Model Reference",
    "title": "NNlib.elu",
    "category": "function",
    "text": "elu(x, α = 1) =\n  x > 0 ? x : α * (exp(x) - 1)\n\nExponential Linear Unit activation function. See Fast and Accurate Deep Network Learning by Exponential Linear Units. You can also specify the coefficient explicitly, e.g. elu(x, 1).\n\n\n\n"
},

{
    "location": "models/layers.html#NNlib.swish",
    "page": "Model Reference",
    "title": "NNlib.swish",
    "category": "function",
    "text": "swish(x) = x * σ(x)\n\nSelf-gated actvation function. See Swish: a Self-Gated Activation Function.\n\n\n\n"
},

{
    "location": "models/layers.html#Activation-Functions-1",
    "page": "Model Reference",
    "title": "Activation Functions",
    "category": "section",
    "text": "Non-linearities that go between layers of your model. Most of these functions are defined in NNlib but are available by default in Flux.Note that, unless otherwise stated, activation functions operate on scalars. To apply them to an array you can call σ.(xs), relu.(xs) and so on.σ\nrelu\nleakyrelu\nelu\nswish"
},

{
    "location": "models/layers.html#Flux.testmode!",
    "page": "Model Reference",
    "title": "Flux.testmode!",
    "category": "function",
    "text": "testmode!(m)\ntestmode!(m, false)\n\nPut layers like Dropout and BatchNorm into testing mode (or back to training mode with false).\n\n\n\n"
},

{
    "location": "models/layers.html#Flux.BatchNorm",
    "page": "Model Reference",
    "title": "Flux.BatchNorm",
    "category": "type",
    "text": "BatchNorm(dims...; λ = identity,\n          initβ = zeros, initγ = ones, ϵ = 1e-8, momentum = .1)\n\nBatch Normalization Layer for Dense layer.\n\nSee Batch Normalization: Accelerating Deep Network Training by Reducing      Internal Covariate Shift\n\nIn the example of MNIST, in order to normalize the input of other layer, put the BatchNorm layer before activation function.\n\nm = Chain(\n  Dense(28^2, 64),\n  BatchNorm(64, λ = relu),\n  Dense(64, 10),\n  BatchNorm(10),\n  softmax)\n\n\n\n"
},

{
    "location": "models/layers.html#Flux.Dropout",
    "page": "Model Reference",
    "title": "Flux.Dropout",
    "category": "type",
    "text": "Dropout(p)\n\nA Dropout layer. For each input, either sets that input to 0 (with probability p) or scales it by 1/(1-p). This is used as a regularisation, i.e. it reduces overfitting during training.\n\nDoes nothing to the input once in testmode!.\n\n\n\n"
},

{
    "location": "models/layers.html#Flux.LayerNorm",
    "page": "Model Reference",
    "title": "Flux.LayerNorm",
    "category": "type",
    "text": "LayerNorm(h::Integer)\n\nA normalisation layer designed to be used with recurrent hidden states of size h. Normalises the mean/stddev of each input before applying a per-neuron gain/bias.\n\n\n\n"
},

{
    "location": "models/layers.html#Normalisation-and-Regularisation-1",
    "page": "Model Reference",
    "title": "Normalisation & Regularisation",
    "category": "section",
    "text": "These layers don\'t affect the structure of the network but may improve training times or reduce overfitting.Flux.testmode!\nBatchNorm\nDropout\nLayerNorm"
},

{
    "location": "training/optimisers.html#",
    "page": "Optimisers",
    "title": "Optimisers",
    "category": "page",
    "text": ""
},

{
    "location": "training/optimisers.html#Optimisers-1",
    "page": "Optimisers",
    "title": "Optimisers",
    "category": "section",
    "text": "Consider a simple linear regression. We create some dummy data, calculate a loss, and backpropagate to calculate gradients for the parameters W and b.W = param(rand(2, 5))\nb = param(rand(2))\n\npredict(x) = W*x .+ b\nloss(x, y) = sum((predict(x) .- y).^2)\n\nx, y = rand(5), rand(2) # Dummy data\nl = loss(x, y) # ~ 3\nback!(l)We want to update each parameter, using the gradient, in order to improve (reduce) the loss. Here\'s one way to do that:function update()\n  η = 0.1 # Learning Rate\n  for p in (W, b)\n    p.data .-= η .* p.grad # Apply the update\n    p.grad .= 0            # Clear the gradient\n  end\nendIf we call update, the parameters W and b will change and our loss should go down.There are two pieces here: one is that we need a list of trainable parameters for the model ([W, b] in this case), and the other is the update step. In this case the update is simply gradient descent (x .-= η .* Δ), but we might choose to do something more advanced, like adding momentum.In this case, getting the variables is trivial, but you can imagine it\'d be more of a pain with some complex stack of layers.m = Chain(\n  Dense(10, 5, σ),\n  Dense(5, 2), softmax)Instead of having to write [m[1].W, m[1].b, ...], Flux provides a params function params(m) that returns a list of all parameters in the model for you.For the update step, there\'s nothing whatsoever wrong with writing the loop above – it\'ll work just fine – but Flux provides various optimisers that make it more convenient.opt = SGD([W, b], 0.1) # Gradient descent with learning rate 0.1\n\nopt() # Carry out the update, modifying `W` and `b`.An optimiser takes a parameter list and returns a function that does the same thing as update above. We can pass either opt or update to our training loop, which will then run the optimiser after every mini-batch of data."
},

{
    "location": "training/optimisers.html#Flux.Optimise.SGD",
    "page": "Optimisers",
    "title": "Flux.Optimise.SGD",
    "category": "function",
    "text": "SGD(params, η = 0.1; decay = 0)\n\nClassic gradient descent optimiser with learning rate η. For each parameter p and its gradient δp, this runs p -= η*δp.\n\nSupports inverse decaying learning rate if the decay argument is provided.\n\n\n\n"
},

{
    "location": "training/optimisers.html#Flux.Optimise.Momentum",
    "page": "Optimisers",
    "title": "Flux.Optimise.Momentum",
    "category": "function",
    "text": "Momentum(params, η = 0.01; ρ = 0.9, decay = 0)\n\nSGD with learning rate  η, momentum ρ and optional learning rate inverse decay.\n\n\n\n"
},

{
    "location": "training/optimisers.html#Flux.Optimise.Nesterov",
    "page": "Optimisers",
    "title": "Flux.Optimise.Nesterov",
    "category": "function",
    "text": "Nesterov(params, η = 0.01; ρ = 0.9, decay = 0)\n\nSGD with learning rate  η, Nesterov momentum ρ and optional learning rate inverse decay.\n\n\n\n"
},

{
    "location": "training/optimisers.html#Flux.Optimise.ADAM",
    "page": "Optimisers",
    "title": "Flux.Optimise.ADAM",
    "category": "function",
    "text": "ADAM(params, η = 0.001; β1 = 0.9, β2 = 0.999, ϵ = 1e-08, decay = 0)\n\nADAM optimiser.\n\n\n\n"
},

{
    "location": "training/optimisers.html#Optimiser-Reference-1",
    "page": "Optimisers",
    "title": "Optimiser Reference",
    "category": "section",
    "text": "All optimisers return a function that, when called, will update the parameters passed to it.SGD\nMomentum\nNesterov\nADAM"
},

{
    "location": "training/training.html#",
    "page": "Training",
    "title": "Training",
    "category": "page",
    "text": ""
},

{
    "location": "training/training.html#Training-1",
    "page": "Training",
    "title": "Training",
    "category": "section",
    "text": "To actually train a model we need three things:A objective function, that evaluates how well a model is doing given some input data.\nA collection of data points that will be provided to the objective function.\nAn optimiser that will update the model parameters appropriately.With these we can call Flux.train!:Flux.train!(objective, data, opt)There are plenty of examples in the model zoo."
},

{
    "location": "training/training.html#Loss-Functions-1",
    "page": "Training",
    "title": "Loss Functions",
    "category": "section",
    "text": "The objective function must return a number representing how far the model is from its target – the loss of the model. The loss function that we defined in basics will work as an objective. We can also define an objective in terms of some model:m = Chain(\n  Dense(784, 32, σ),\n  Dense(32, 10), softmax)\n\nloss(x, y) = Flux.mse(m(x), y)\n\n# later\nFlux.train!(loss, data, opt)The objective will almost always be defined in terms of some cost function that measures the distance of the prediction m(x) from the target y. Flux has several of these built in, like mse for mean squared error or crossentropy for cross entropy loss, but you can calculate it however you want."
},

{
    "location": "training/training.html#Datasets-1",
    "page": "Training",
    "title": "Datasets",
    "category": "section",
    "text": "The data argument provides a collection of data to train with (usually a set of inputs x and target outputs y). For example, here\'s a dummy data set with only one data point:x = rand(784)\ny = rand(10)\ndata = [(x, y)]Flux.train! will call loss(x, y), calculate gradients, update the weights and then move on to the next data point if there is one. We can train the model on the same data three times:data = [(x, y), (x, y), (x, y)]\n# Or equivalently\ndata = Iterators.repeated((x, y), 3)It\'s common to load the xs and ys separately. In this case you can use zip:xs = [rand(784), rand(784), rand(784)]\nys = [rand( 10), rand( 10), rand( 10)]\ndata = zip(xs, ys)Note that, by default, train! only loops over the data once (a single \"epoch\"). A convenient way to run multiple epochs from the REPL is provided by @epochs.julia> using Flux: @epochs\n\njulia> @epochs 2 println(\"hello\")\nINFO: Epoch 1\nhello\nINFO: Epoch 2\nhello\n\njulia> @epochs 2 Flux.train!(...)\n# Train for two epochs"
},

{
    "location": "training/training.html#Callbacks-1",
    "page": "Training",
    "title": "Callbacks",
    "category": "section",
    "text": "train! takes an additional argument, cb, that\'s used for callbacks so that you can observe the training process. For example:train!(objective, data, opt, cb = () -> println(\"training\"))Callbacks are called for every batch of training data. You can slow this down using Flux.throttle(f, timeout) which prevents f from being called more than once every timeout seconds.A more typical callback might look like this:test_x, test_y = # ... create single batch of test data ...\nevalcb() = @show(loss(test_x, test_y))\n\nFlux.train!(objective, data, opt,\n            cb = throttle(evalcb, 5))"
},

{
    "location": "data/onehot.html#",
    "page": "One-Hot Encoding",
    "title": "One-Hot Encoding",
    "category": "page",
    "text": ""
},

{
    "location": "data/onehot.html#One-Hot-Encoding-1",
    "page": "One-Hot Encoding",
    "title": "One-Hot Encoding",
    "category": "section",
    "text": "It\'s common to encode categorical variables (like true, false or cat, dog) in \"one-of-k\" or \"one-hot\" form. Flux provides the onehot function to make this easy.julia> using Flux: onehot\n\njulia> onehot(:b, [:a, :b, :c])\n3-element Flux.OneHotVector:\n false\n  true\n false\n\njulia> onehot(:c, [:a, :b, :c])\n3-element Flux.OneHotVector:\n false\n false\n  trueThe inverse is argmax (which can take a general probability distribution, as well as just booleans).julia> argmax(ans, [:a, :b, :c])\n:c\n\njulia> argmax([true, false, false], [:a, :b, :c])\n:a\n\njulia> argmax([0.3, 0.2, 0.5], [:a, :b, :c])\n:c"
},

{
    "location": "data/onehot.html#Batches-1",
    "page": "One-Hot Encoding",
    "title": "Batches",
    "category": "section",
    "text": "onehotbatch creates a batch (matrix) of one-hot vectors, and argmax treats matrices as batches.julia> using Flux: onehotbatch\n\njulia> onehotbatch([:b, :a, :b], [:a, :b, :c])\n3×3 Flux.OneHotMatrix:\n false   true  false\n  true  false   true\n false  false  false\n\njulia> onecold(ans, [:a, :b, :c])\n3-element Array{Symbol,1}:\n  :b\n  :a\n  :bNote that these operations returned OneHotVector and OneHotMatrix rather than Arrays. OneHotVectors behave like normal vectors but avoid any unnecessary cost compared to using an integer index directly. For example, multiplying a matrix with a one-hot vector simply slices out the relevant row of the matrix under the hood."
},

{
    "location": "gpu.html#",
    "page": "GPU Support",
    "title": "GPU Support",
    "category": "page",
    "text": ""
},

{
    "location": "gpu.html#GPU-Support-1",
    "page": "GPU Support",
    "title": "GPU Support",
    "category": "section",
    "text": "Support for array operations on other hardware backends, like GPUs, is provided by external packages like CuArrays. Flux is agnostic to array types, so we simply need to move model weights and data to the GPU and Flux will handle it.For example, we can use CuArrays (with the cu converter) to run our basic example on an NVIDIA GPU.using CuArrays\n\nW = cu(rand(2, 5)) # a 2×5 CuArray\nb = cu(rand(2))\n\npredict(x) = W*x .+ b\nloss(x, y) = sum((predict(x) .- y).^2)\n\nx, y = cu(rand(5)), cu(rand(2)) # Dummy data\nloss(x, y) # ~ 3Note that we convert both the parameters (W, b) and the data set (x, y) to cuda arrays. Taking derivatives and training works exactly as before.If you define a structured model, like a Dense layer or Chain, you just need to convert the internal parameters. Flux provides mapleaves, which allows you to alter all parameters of a model at once.d = Dense(10, 5, σ)\nd = mapleaves(cu, d)\nd.W # Tracked CuArray\nd(cu(rand(10))) # CuArray output\n\nm = Chain(Dense(10, 5, σ), Dense(5, 2), softmax)\nm = mapleaves(cu, m)\nd(cu(rand(10)))As a convenience, Flux provides the gpu function to convert models and data to the GPU if one is available. By default, it\'ll do nothing, but loading CuArrays will cause it to move data to the GPU instead.julia> using Flux, CuArrays\n\njulia> m = Dense(10,5) |> gpu\nDense(10, 5)\n\njulia> x = rand(10) |> gpu\n10-element CuArray{Float32,1}:\n 0.800225\n ⋮\n 0.511655\n\njulia> m(x)\nTracked 5-element CuArray{Float32,1}:\n -0.30535\n ⋮\n -0.618002The analogue cpu is also available for moving models and data back off of the GPU.``` julia> x = rand(10) |> gpu 10-element CuArray{Float32,1}:  0.235164  ⋮  0.192538julia> x |> cpu 10-element Array{Float32,1}:  0.235164  ⋮  0.192538  ```"
},

{
    "location": "saving.html#",
    "page": "Saving & Loading",
    "title": "Saving & Loading",
    "category": "page",
    "text": ""
},

{
    "location": "saving.html#Saving-and-Loading-Models-1",
    "page": "Saving & Loading",
    "title": "Saving and Loading Models",
    "category": "section",
    "text": "You may wish to save models so that they can be loaded and run in a later session. The easiest way to do this is via BSON.jl.Save a model:julia> using Flux\n\njulia> model = Chain(Dense(10,5,relu),Dense(5,2),softmax)\nChain(Dense(10, 5, NNlib.relu), Dense(5, 2), NNlib.softmax)\n\njulia> using BSON: @save\n\njulia> @save \"mymodel.bson\" modelLoad it again:julia> using Flux\n\njulia> using BSON: @load\n\njulia> @load \"mymodel.bson\" model\n\njulia> model\nChain(Dense(10, 5, NNlib.relu), Dense(5, 2), NNlib.softmax)Models are just normal Julia structs, so it\'s fine to use any Julia storage format for this purpose. BSON.jl is particularly well supported and most likely to be forwards compatible (that is, models saved now will load in future versions of Flux).note: Note\nIf a saved model\'s weights are stored on the GPU, the model will not load later on if there is no GPU support available. It\'s best to move your model to the CPU with cpu(model) before saving it."
},

{
    "location": "saving.html#Saving-Model-Weights-1",
    "page": "Saving & Loading",
    "title": "Saving Model Weights",
    "category": "section",
    "text": "In some cases it may be useful to save only the model parameters themselves, and rebuild the model architecture in your code. You can use params(model) to get model parameters. You can also use data.(params) to remove tracking.julia> using Flux\n\njulia> model = Chain(Dense(10,5,relu),Dense(5,2),softmax)\nChain(Dense(10, 5, NNlib.relu), Dense(5, 2), NNlib.softmax)\n\njulia> weights = Tracker.data.(params(model));\n\njulia> using BSON: @save\n\njulia> @save \"mymodel.bson\" weightsYou can easily load parameters back into a model with Flux.loadparams!.julia> using Flux\n\njulia> model = Chain(Dense(10,5,relu),Dense(5,2),softmax)\nChain(Dense(10, 5, NNlib.relu), Dense(5, 2), NNlib.softmax)\n\njulia> using BSON: @load\n\njulia> @load \"mymodel.bson\" weights\n\njulia> Flux.loadparams!(model, weights)The new model we created will now be identical to the one we saved parameters for."
},

{
    "location": "saving.html#Checkpointing-1",
    "page": "Saving & Loading",
    "title": "Checkpointing",
    "category": "section",
    "text": "In longer training runs it\'s a good idea to periodically save your model, so that you can resume if training is interrupted (for example, if there\'s a power cut). You can do this by saving the model in the callback provided to train!.using Flux: throttle\nusing BSON: @save\n\nm = Chain(Dense(10,5,relu),Dense(5,2),softmax)\n\nevalcb = throttle(30) do\n  # Show loss\n  @save \"model-checkpoint.bson\" model\nendThis will update the \"model-checkpoint.bson\" file every thirty seconds.You can get more advanced by saving a series of models throughout training, for example@save \"model-$(now()).bson\" modelwill produce a series of models like \"model-2018-03-06T02:57:10.41.bson\". You could also store the current test set loss, so that it\'s easy to (for example) revert to an older copy of the model if it starts to overfit.@save \"model-$(now()).bson\" model loss = testloss()You can even store optimiser state alongside the model, to resume training exactly where you left off.opt = ADAM(params(model))\n@save \"model-$(now()).bson\" model opt"
},

{
    "location": "community.html#",
    "page": "Community",
    "title": "Community",
    "category": "page",
    "text": ""
},

{
    "location": "community.html#Community-1",
    "page": "Community",
    "title": "Community",
    "category": "section",
    "text": "All Flux users are welcome to join our community on the Julia forum, the slack (channel #machine-learning), or Flux\'s Gitter. If you have questions or issues we\'ll try to help you out.If you\'re interested in hacking on Flux, the source code is open and easy to understand – it\'s all just the same Julia code you work with normally. You might be interested in our intro issues to get started."
},

]}
