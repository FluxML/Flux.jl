var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#Flux:-The-Julia-Machine-Learning-Library-1",
    "page": "Home",
    "title": "Flux: The Julia Machine Learning Library",
    "category": "section",
    "text": "Flux is a library for machine learning. It comes \"batteries-included\" with many useful tools built in, but also lets you use the full power of the Julia language where you need it. We follow a few key principles:Doing the obvious thing. Flux has relatively few explicit APIs for features like regularisation or embeddings. Instead, writing down the mathematical form will work – and be fast.\nYou could have written Flux. All of it, from LSTMs to GPU kernels, is straightforward Julia code. When in doubt, it’s well worth looking at the source. If you need something different, you can easily roll your own.\nPlay nicely with others. Flux works well with Julia libraries from data frames and images to differential equation solvers, so you can easily build complex data processing pipelines that integrate Flux models."
},

{
    "location": "#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "Download Julia 1.0 or later, if you haven\'t already. You can add Flux from using Julia\'s package manager, by typing ] add Flux in the Julia prompt.If you have CUDA you can also run ] add CuArrays to get GPU support; see here for more details."
},

{
    "location": "#Learning-Flux-1",
    "page": "Home",
    "title": "Learning Flux",
    "category": "section",
    "text": "There are several different ways to learn Flux. If you just want to get started writing models, the model zoo gives good starting points for many common ones. This documentation provides a reference to all of Flux\'s APIs, as well as a from-scratch introduction to Flux\'s take on models and how they work. Once you understand these docs, congratulations, you also understand Flux\'s source code, which is intended to be concise, legible and a good reference for more advanced concepts."
},

{
    "location": "models/basics/#",
    "page": "Basics",
    "title": "Basics",
    "category": "page",
    "text": ""
},

{
    "location": "models/basics/#Model-Building-Basics-1",
    "page": "Basics",
    "title": "Model-Building Basics",
    "category": "section",
    "text": ""
},

{
    "location": "models/basics/#Taking-Gradients-1",
    "page": "Basics",
    "title": "Taking Gradients",
    "category": "section",
    "text": "Flux\'s core feature is taking gradients of Julia code. The gradient function takes another Julia function f and a set of arguments, and returns the gradient with respect to each argument. (It\'s a good idea to try pasting these examples in the Julia terminal.)julia> using Flux.Tracker\n\njulia> f(x) = 3x^2 + 2x + 1;\n\njulia> df(x) = Tracker.gradient(f, x; nest = true)[1]; # df/dx = 6x + 2\n\njulia> df(2)\n14.0 (tracked)\n\njulia> d2f(x) = Tracker.gradient(df, x; nest = true)[1]; # d²f/dx² = 6\n\njulia> d2f(2)\n6.0 (tracked)(We\'ll learn more about why these numbers show up as (tracked) below.)When a function has many parameters, we can pass them all in explicitly:julia> f(W, b, x) = W * x + b;\n\njulia> Tracker.gradient(f, 2, 3, 4)\n(4.0 (tracked), 1.0 (tracked), 2.0 (tracked))But machine learning models can have hundreds of parameters! Flux offers a nice way to handle this. We can tell Flux to treat something as a parameter via param. Then we can collect these together and tell gradient to collect the gradients of all params at once.julia> using Flux\n\njulia> W = param(2) \n2.0 (tracked)\n\njulia> b = param(3)\n3.0 (tracked)\n\njulia> f(x) = W * x + b;\n\njulia> grads = Tracker.gradient(() -> f(4), params(W, b));\n\njulia> grads[W]\n4.0\n\njulia> grads[b]\n1.0There are a few things to notice here. Firstly, W and b now show up as tracked. Tracked things behave like normal numbers or arrays, but keep records of everything you do with them, allowing Flux to calculate their gradients. gradient takes a zero-argument function; no arguments are necessary because the params tell it what to differentiate.This will come in really handy when dealing with big, complicated models. For now, though, let\'s start with something simple."
},

{
    "location": "models/basics/#Simple-Models-1",
    "page": "Basics",
    "title": "Simple Models",
    "category": "section",
    "text": "Consider a simple linear regression, which tries to predict an output array y from an input x.W = rand(2, 5)\nb = rand(2)\n\npredict(x) = W*x .+ b\n\nfunction loss(x, y)\n  ŷ = predict(x)\n  sum((y .- ŷ).^2)\nend\n\nx, y = rand(5), rand(2) # Dummy data\nloss(x, y) # ~ 3To improve the prediction we can take the gradients of W and b with respect to the loss and perform gradient descent. Let\'s tell Flux that W and b are parameters, just like we did above.using Flux.Tracker\n\nW = param(W)\nb = param(b)\n\ngs = Tracker.gradient(() -> loss(x, y), params(W, b))Now that we have gradients, we can pull them out and update W to train the model. The update!(W, Δ) function applies W = W + Δ, which we can use for gradient descent.using Flux.Tracker: update!\n\nΔ = gs[W]\n\n# Update the parameter and reset the gradient\nupdate!(W, -0.1Δ)\n\nloss(x, y) # ~ 2.5The loss has decreased a little, meaning that our prediction x is closer to the target y. If we have some data we can already try training the model.All deep learning in Flux, however complex, is a simple generalisation of this example. Of course, models can look very different – they might have millions of parameters or complex control flow. Let\'s see how Flux handles more complex models."
},

{
    "location": "models/basics/#Building-Layers-1",
    "page": "Basics",
    "title": "Building Layers",
    "category": "section",
    "text": "It\'s common to create more complex models than the linear regression above. For example, we might want to have two linear layers with a nonlinearity like sigmoid (σ) in between them. In the above style we could write this as:using Flux\n\nW1 = param(rand(3, 5))\nb1 = param(rand(3))\nlayer1(x) = W1 * x .+ b1\n\nW2 = param(rand(2, 3))\nb2 = param(rand(2))\nlayer2(x) = W2 * x .+ b2\n\nmodel(x) = layer2(σ.(layer1(x)))\n\nmodel(rand(5)) # => 2-element vectorThis works but is fairly unwieldy, with a lot of repetition – especially as we add more layers. One way to factor this out is to create a function that returns linear layers.function linear(in, out)\n  W = param(randn(out, in))\n  b = param(randn(out))\n  x -> W * x .+ b\nend\n\nlinear1 = linear(5, 3) # we can access linear1.W etc\nlinear2 = linear(3, 2)\n\nmodel(x) = linear2(σ.(linear1(x)))\n\nmodel(rand(5)) # => 2-element vectorAnother (equivalent) way is to create a struct that explicitly represents the affine layer.struct Affine\n  W\n  b\nend\n\nAffine(in::Integer, out::Integer) =\n  Affine(param(randn(out, in)), param(randn(out)))\n\n# Overload call, so the object can be used as a function\n(m::Affine)(x) = m.W * x .+ m.b\n\na = Affine(10, 5)\n\na(rand(10)) # => 5-element vectorCongratulations! You just built the Dense layer that comes with Flux. Flux has many interesting layers available, but they\'re all things you could have built yourself very easily.(There is one small difference with Dense – for convenience it also takes an activation function, like Dense(10, 5, σ).)"
},

{
    "location": "models/basics/#Stacking-It-Up-1",
    "page": "Basics",
    "title": "Stacking It Up",
    "category": "section",
    "text": "It\'s pretty common to write models that look something like:layer1 = Dense(10, 5, σ)\n# ...\nmodel(x) = layer3(layer2(layer1(x)))For long chains, it might be a bit more intuitive to have a list of layers, like this:using Flux\n\nlayers = [Dense(10, 5, σ), Dense(5, 2), softmax]\n\nmodel(x) = foldl((x, m) -> m(x), layers, init = x)\n\nmodel(rand(10)) # => 2-element vectorHandily, this is also provided for in Flux:model2 = Chain(\n  Dense(10, 5, σ),\n  Dense(5, 2),\n  softmax)\n\nmodel2(rand(10)) # => 2-element vectorThis quickly starts to look like a high-level deep learning library; yet you can see how it falls out of simple abstractions, and we lose none of the power of Julia code.A nice property of this approach is that because \"models\" are just functions (possibly with trainable parameters), you can also see this as simple function composition.m = Dense(5, 2) ∘ Dense(10, 5, σ)\n\nm(rand(10))Likewise, Chain will happily work with any Julia function.m = Chain(x -> x^2, x -> x+1)\n\nm(5) # => 26"
},

{
    "location": "models/basics/#Layer-helpers-1",
    "page": "Basics",
    "title": "Layer helpers",
    "category": "section",
    "text": "Flux provides a set of helpers for custom layers, which you can enable by callingFlux.@treelike AffineThis enables a useful extra set of functionality for our Affine layer, such as collecting its parameters or moving it to the GPU."
},

{
    "location": "models/recurrence/#",
    "page": "Recurrence",
    "title": "Recurrence",
    "category": "page",
    "text": ""
},

{
    "location": "models/recurrence/#Recurrent-Models-1",
    "page": "Recurrence",
    "title": "Recurrent Models",
    "category": "section",
    "text": ""
},

{
    "location": "models/recurrence/#Recurrent-Cells-1",
    "page": "Recurrence",
    "title": "Recurrent Cells",
    "category": "section",
    "text": "In the simple feedforward case, our model m is a simple function from various inputs xᵢ to predictions yᵢ. (For example, each x might be an MNIST digit and each y a digit label.) Each prediction is completely independent of any others, and using the same x will always produce the same y.y₁ = f(x₁)\ny₂ = f(x₂)\ny₃ = f(x₃)\n# ...Recurrent networks introduce a hidden state that gets carried over each time we run the model. The model now takes the old h as an input, and produces a new h as output, each time we run it.h = # ... initial state ...\nh, y₁ = f(h, x₁)\nh, y₂ = f(h, x₂)\nh, y₃ = f(h, x₃)\n# ...Information stored in h is preserved for the next prediction, allowing it to function as a kind of memory. This also means that the prediction made for a given x depends on all the inputs previously fed into the model.(This might be important if, for example, each x represents one word of a sentence; the model\'s interpretation of the word \"bank\" should change if the previous input was \"river\" rather than \"investment\".)Flux\'s RNN support closely follows this mathematical perspective. The most basic RNN is as close as possible to a standard Dense layer, and the output is also the hidden state.Wxh = randn(5, 10)\nWhh = randn(5, 5)\nb   = randn(5)\n\nfunction rnn(h, x)\n  h = tanh.(Wxh * x .+ Whh * h .+ b)\n  return h, h\nend\n\nx = rand(10) # dummy data\nh = rand(5)  # initial hidden state\n\nh, y = rnn(h, x)If you run the last line a few times, you\'ll notice the output y changing slightly even though the input x is the same.We sometimes refer to functions like rnn above, which explicitly manage state, as recurrent cells. There are various recurrent cells available, which are documented in the layer reference. The hand-written example above can be replaced with:using Flux\n\nrnn2 = Flux.RNNCell(10, 5)\n\nx = rand(10) # dummy data\nh = rand(5)  # initial hidden state\n\nh, y = rnn2(h, x)"
},

{
    "location": "models/recurrence/#Stateful-Models-1",
    "page": "Recurrence",
    "title": "Stateful Models",
    "category": "section",
    "text": "For the most part, we don\'t want to manage hidden states ourselves, but to treat our models as being stateful. Flux provides the Recur wrapper to do this.x = rand(10)\nh = rand(5)\n\nm = Flux.Recur(rnn, h)\n\ny = m(x)The Recur wrapper stores the state between runs in the m.state field.If you use the RNN(10, 5) constructor – as opposed to RNNCell – you\'ll see that it\'s simply a wrapped cell.julia> RNN(10, 5)\nRecur(RNNCell(Dense(15, 5)))"
},

{
    "location": "models/recurrence/#Sequences-1",
    "page": "Recurrence",
    "title": "Sequences",
    "category": "section",
    "text": "Often we want to work with sequences of inputs, rather than individual xs.seq = [rand(10) for i = 1:10]With Recur, applying our model to each element of a sequence is trivial:m.(seq) # returns a list of 5-element vectorsThis works even when we\'ve chain recurrent layers into a larger model.m = Chain(LSTM(10, 15), Dense(15, 5))\nm.(seq)"
},

{
    "location": "models/recurrence/#Truncating-Gradients-1",
    "page": "Recurrence",
    "title": "Truncating Gradients",
    "category": "section",
    "text": "By default, calculating the gradients in a recurrent layer involves its entire history. For example, if we call the model on 100 inputs, we\'ll have to calculate the gradient for those 100 calls. If we then calculate another 10 inputs we have to calculate 110 gradients – this accumulates and quickly becomes expensive.To avoid this we can truncate the gradient calculation, forgetting the history.truncate!(m)Calling truncate! wipes the slate clean, so we can call the model with more inputs without building up an expensive gradient computation.truncate! makes sense when you are working with multiple chunks of a large sequence, but we may also want to work with a set of independent sequences. In this case the hidden state should be completely reset to its original value, throwing away any accumulated information. reset! does this for you."
},

{
    "location": "models/regularisation/#",
    "page": "Regularisation",
    "title": "Regularisation",
    "category": "page",
    "text": ""
},

{
    "location": "models/regularisation/#Regularisation-1",
    "page": "Regularisation",
    "title": "Regularisation",
    "category": "section",
    "text": "Applying regularisation to model parameters is straightforward. We just need to apply an appropriate regulariser, such as norm, to each model parameter and add the result to the overall loss.For example, say we have a simple regression.using Flux: crossentropy\nm = Dense(10, 5)\nloss(x, y) = crossentropy(softmax(m(x)), y)We can regularise this by taking the (L2) norm of the parameters, m.W and m.b.penalty() = norm(m.W) + norm(m.b)\nloss(x, y) = crossentropy(softmax(m(x)), y) + penalty()When working with layers, Flux provides the params function to grab all parameters at once. We can easily penalise everything with sum(norm, params).julia> params(m)\n2-element Array{Any,1}:\n param([0.355408 0.533092; … 0.430459 0.171498])\n param([0.0, 0.0, 0.0, 0.0, 0.0])\n\njulia> sum(norm, params(m))\n26.01749952921026 (tracked)Here\'s a larger example with a multi-layer perceptron.m = Chain(\n  Dense(28^2, 128, relu),\n  Dense(128, 32, relu),\n  Dense(32, 10), softmax)\n\nloss(x, y) = crossentropy(m(x), y) + sum(norm, params(m))\n\nloss(rand(28^2), rand(10))One can also easily add per-layer regularisation via the activations function:julia> c = Chain(Dense(10,5,σ),Dense(5,2),softmax)\nChain(Dense(10, 5, NNlib.σ), Dense(5, 2), NNlib.softmax)\n\njulia> activations(c, rand(10))\n3-element Array{Any,1}:\n param([0.71068, 0.831145, 0.751219, 0.227116, 0.553074])\n param([0.0330606, -0.456104])\n param([0.61991, 0.38009])\n\njulia> sum(norm, ans)\n2.639678767773633 (tracked)"
},

{
    "location": "models/layers/#",
    "page": "Model Reference",
    "title": "Model Reference",
    "category": "page",
    "text": ""
},

{
    "location": "models/layers/#Flux.Chain",
    "page": "Model Reference",
    "title": "Flux.Chain",
    "category": "type",
    "text": "Chain(layers...)\n\nChain multiple layers / functions together, so that they are called in sequence on a given input.\n\nm = Chain(x -> x^2, x -> x+1)\nm(5) == 26\n\nm = Chain(Dense(10, 5), Dense(5, 2))\nx = rand(10)\nm(x) == m[2](m[1](x))\n\nChain also supports indexing and slicing, e.g. m[2] or m[1:end-1]. m[1:3](x) will calculate the output of the first three layers.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Flux.Dense",
    "page": "Model Reference",
    "title": "Flux.Dense",
    "category": "type",
    "text": "Dense(in::Integer, out::Integer, σ = identity)\n\nCreates a traditional Dense layer with parameters W and b.\n\ny = σ.(W * x .+ b)\n\nThe input x must be a vector of length in, or a batch of vectors represented as an in × N matrix. The out y will be a vector or batch of length out.\n\njulia> d = Dense(5, 2)\nDense(5, 2)\n\njulia> d(rand(5))\nTracked 2-element Array{Float64,1}:\n  0.00257447\n  -0.00449443\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Basic-Layers-1",
    "page": "Model Reference",
    "title": "Basic Layers",
    "category": "section",
    "text": "These core layers form the foundation of almost all neural networks.Chain\nDense"
},

{
    "location": "models/layers/#Flux.Conv",
    "page": "Model Reference",
    "title": "Flux.Conv",
    "category": "type",
    "text": "Conv(size, in=>out)\nConv(size, in=>out, relu)\n\nStandard convolutional layer. size should be a tuple like (2, 2). in and out specify the number of input and output channels respectively.\n\nExample: Applying Conv layer to a 1-channel input using a 2x2 window size,          giving us a 16-channel output. Output is activated with ReLU.\n\nsize = (2,2)\nin = 1\nout = 16 \nConv((2, 2), 1=>16, relu)\n\nData should be stored in WHCN order (width, height, # channels, # batches).  In other words, a 100×100 RGB image would be a 100×100×3×1 array,  and a batch of 50 would be a 100×100×3×50 array.\n\nTakes the keyword arguments pad, stride and dilation.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Flux.MaxPool",
    "page": "Model Reference",
    "title": "Flux.MaxPool",
    "category": "type",
    "text": "MaxPool(k)\n\nMax pooling layer. k stands for the size of the window for each dimension of the input.\n\nTakes the keyword arguments pad and stride.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Flux.MeanPool",
    "page": "Model Reference",
    "title": "Flux.MeanPool",
    "category": "type",
    "text": "MeanPool(k)\n\nMean pooling layer. k stands for the size of the window for each dimension of the input.\n\nTakes the keyword arguments pad and stride.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Flux.DepthwiseConv",
    "page": "Model Reference",
    "title": "Flux.DepthwiseConv",
    "category": "type",
    "text": "DepthwiseConv(size, in)\nDepthwiseConv(size, in=>mul)\nDepthwiseConv(size, in=>mul, relu)\n\nDepthwise convolutional layer. size should be a tuple like (2, 2). in and mul specify the number of input channels and channel multiplier respectively. In case the mul is not specified it is taken as 1.\n\nData should be stored in WHCN order. In other words, a 100×100 RGB image would be a 100×100×3 array, and a batch of 50 would be a 100×100×3×50 array.\n\nTakes the keyword arguments pad and stride.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Flux.ConvTranspose",
    "page": "Model Reference",
    "title": "Flux.ConvTranspose",
    "category": "type",
    "text": "ConvTranspose(size, in=>out)\nConvTranspose(size, in=>out, relu)\n\nStandard convolutional transpose layer. size should be a tuple like (2, 2). in and out specify the number of input and output channels respectively. Data should be stored in WHCN order. In other words, a 100×100 RGB image would be a 100×100×3 array, and a batch of 50 would be a 100×100×3×50 array. Takes the keyword arguments pad, stride and dilation.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Convolution-and-Pooling-Layers-1",
    "page": "Model Reference",
    "title": "Convolution and Pooling Layers",
    "category": "section",
    "text": "These layers are used to build convolutional neural networks (CNNs).Conv\nMaxPool\nMeanPool\nDepthwiseConv\nConvTranspose"
},

{
    "location": "models/layers/#Flux.RNN",
    "page": "Model Reference",
    "title": "Flux.RNN",
    "category": "function",
    "text": "RNN(in::Integer, out::Integer, σ = tanh)\n\nThe most basic recurrent layer; essentially acts as a Dense layer, but with the output fed back into the input each time step.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Flux.LSTM",
    "page": "Model Reference",
    "title": "Flux.LSTM",
    "category": "function",
    "text": "LSTM(in::Integer, out::Integer)\n\nLong Short Term Memory recurrent layer. Behaves like an RNN but generally exhibits a longer memory span over sequences.\n\nSee this article for a good overview of the internals.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Flux.GRU",
    "page": "Model Reference",
    "title": "Flux.GRU",
    "category": "function",
    "text": "GRU(in::Integer, out::Integer)\n\nGated Recurrent Unit layer. Behaves like an RNN but generally exhibits a longer memory span over sequences.\n\nSee this article for a good overview of the internals.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Flux.Recur",
    "page": "Model Reference",
    "title": "Flux.Recur",
    "category": "type",
    "text": "Recur(cell)\n\nRecur takes a recurrent cell and makes it stateful, managing the hidden state in the background. cell should be a model of the form:\n\nh, y = cell(h, x...)\n\nFor example, here\'s a recurrent network that keeps a running total of its inputs.\n\naccum(h, x) = (h+x, x)\nrnn = Flux.Recur(accum, 0)\nrnn(2) # 2\nrnn(3) # 3\nrnn.state # 5\nrnn.(1:10) # apply to a sequence\nrnn.state # 60\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Recurrent-Layers-1",
    "page": "Model Reference",
    "title": "Recurrent Layers",
    "category": "section",
    "text": "Much like the core layers above, but can be used to process sequence data (as well as other kinds of structured data).RNN\nLSTM\nGRU\nFlux.Recur"
},

{
    "location": "models/layers/#Flux.Maxout",
    "page": "Model Reference",
    "title": "Flux.Maxout",
    "category": "type",
    "text": "Maxout(over)\n\nMaxout is a neural network layer, which has a number of internal layers, which all have the same input, and the maxout returns the elementwise maximium of the internal layers\' outputs.\n\nMaxout over linear dense layers satisfies the univeral approximation theorem.\n\nReference: Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, and Yoshua Bengio.\n\nMaxout networks.\n\nIn Proceedings of the 30th International Conference on International Conference on Machine Learning - Volume 28 (ICML\'13), Sanjoy Dasgupta and David McAllester (Eds.), Vol. 28. JMLR.org III-1319-III-1327. https://arxiv.org/pdf/1302.4389.pdf\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Other-General-Purpose-Layers-1",
    "page": "Model Reference",
    "title": "Other General Purpose Layers",
    "category": "section",
    "text": "These are marginally more obscure than the Basic Layers. But in contrast to the layers described in the other sections are not readily grouped around a particular purpose (e.g. CNNs or RNNs).Maxout"
},

{
    "location": "models/layers/#Flux.testmode!",
    "page": "Model Reference",
    "title": "Flux.testmode!",
    "category": "function",
    "text": "testmode!(m)\ntestmode!(m, false)\n\nPut layers like Dropout and BatchNorm into testing mode (or back to training mode with false).\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Flux.BatchNorm",
    "page": "Model Reference",
    "title": "Flux.BatchNorm",
    "category": "type",
    "text": "BatchNorm(channels::Integer, σ = identity;\n          initβ = zeros, initγ = ones,\n          ϵ = 1e-8, momentum = .1)\n\nBatch Normalization layer. The channels input should be the size of the channel dimension in your data (see below).\n\nGiven an array with N dimensions, call the N-1th the channel dimension. (For a batch of feature vectors this is just the data dimension, for WHCN images it\'s the usual channel dimension.)\n\nBatchNorm computes the mean and variance for each each W×H×1×N slice and shifts them to have a new mean and variance (corresponding to the learnable, per-channel bias and scale parameters).\n\nSee Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.\n\nExample:\n\nm = Chain(\n  Dense(28^2, 64),\n  BatchNorm(64, relu),\n  Dense(64, 10),\n  BatchNorm(10),\n  softmax)\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Flux.Dropout",
    "page": "Model Reference",
    "title": "Flux.Dropout",
    "category": "type",
    "text": "Dropout(p)\n\nA Dropout layer. For each input, either sets that input to 0 (with probability p) or scales it by 1/(1-p). This is used as a regularisation, i.e. it reduces overfitting during training.\n\nDoes nothing to the input once in testmode!.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Flux.LayerNorm",
    "page": "Model Reference",
    "title": "Flux.LayerNorm",
    "category": "type",
    "text": "LayerNorm(h::Integer)\n\nA normalisation layer designed to be used with recurrent hidden states of size h. Normalises the mean/stddev of each input before applying a per-neuron gain/bias.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Normalisation-and-Regularisation-1",
    "page": "Model Reference",
    "title": "Normalisation & Regularisation",
    "category": "section",
    "text": "These layers don\'t affect the structure of the network but may improve training times or reduce overfitting.Flux.testmode!\nBatchNorm\nDropout\nLayerNorm"
},

{
    "location": "models/layers/#NNlib.σ",
    "page": "Model Reference",
    "title": "NNlib.σ",
    "category": "function",
    "text": "σ(x) = 1 / (1 + exp(-x))\n\nClassic sigmoid activation function.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#NNlib.relu",
    "page": "Model Reference",
    "title": "NNlib.relu",
    "category": "function",
    "text": "relu(x) = max(0, x)\n\nRectified Linear Unit activation function.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#NNlib.leakyrelu",
    "page": "Model Reference",
    "title": "NNlib.leakyrelu",
    "category": "function",
    "text": "leakyrelu(x) = max(0.01x, x)\n\nLeaky Rectified Linear Unit activation function. You can also specify the coefficient explicitly, e.g. leakyrelu(x, 0.01).\n\n\n\n\n\n"
},

{
    "location": "models/layers/#NNlib.elu",
    "page": "Model Reference",
    "title": "NNlib.elu",
    "category": "function",
    "text": "elu(x, α = 1) =\n  x > 0 ? x : α * (exp(x) - 1)\n\nExponential Linear Unit activation function. See Fast and Accurate Deep Network Learning by Exponential Linear Units. You can also specify the coefficient explicitly, e.g. elu(x, 1).\n\n\n\n\n\n"
},

{
    "location": "models/layers/#NNlib.swish",
    "page": "Model Reference",
    "title": "NNlib.swish",
    "category": "function",
    "text": "swish(x) = x * σ(x)\n\nSelf-gated actvation function. See Swish: a Self-Gated Activation Function.\n\n\n\n\n\n"
},

{
    "location": "models/layers/#Activation-Functions-1",
    "page": "Model Reference",
    "title": "Activation Functions",
    "category": "section",
    "text": "Non-linearities that go between layers of your model. Most of these functions are defined in NNlib but are available by default in Flux.Note that, unless otherwise stated, activation functions operate on scalars. To apply them to an array you can call σ.(xs), relu.(xs) and so on.σ\nrelu\nleakyrelu\nelu\nswish"
},

{
    "location": "models/layers/#Normalisation-and-Regularisation-2",
    "page": "Model Reference",
    "title": "Normalisation & Regularisation",
    "category": "section",
    "text": "These layers don\'t affect the structure of the network but may improve training times or reduce overfitting.Flux.testmode!\nBatchNorm\nDropout\nAlphaDropout\nLayerNorm"
},

{
    "location": "training/optimisers/#",
    "page": "Optimisers",
    "title": "Optimisers",
    "category": "page",
    "text": ""
},

{
    "location": "training/optimisers/#Optimisers-1",
    "page": "Optimisers",
    "title": "Optimisers",
    "category": "section",
    "text": "Consider a simple linear regression. We create some dummy data, calculate a loss, and backpropagate to calculate gradients for the parameters W and b.using Flux, Flux.Tracker\n\nW = param(rand(2, 5))\nb = param(rand(2))\n\npredict(x) = W*x .+ b\nloss(x, y) = sum((predict(x) .- y).^2)\n\nx, y = rand(5), rand(2) # Dummy data\nl = loss(x, y) # ~ 3\n\nθ = Params([W, b])\ngrads = Tracker.gradient(() -> loss(x, y), θ)We want to update each parameter, using the gradient, in order to improve (reduce) the loss. Here\'s one way to do that:using Flux.Tracker: grad, update!\n\nη = 0.1 # Learning Rate\nfor p in (W, b)\n  update!(p, -η * grads[p])\nendRunning this will alter the parameters W and b and our loss should go down. Flux provides a more general way to do optimiser updates like this.opt = Descent(0.1) # Gradient descent with learning rate 0.1\n\nfor p in (W, b)\n  update!(opt, p, grads[p])\nendAn optimiser update! accepts a parameter and a gradient, and updates the parameter according to the chosen rule. We can also pass opt to our training loop, which will update all parameters of the model in a loop. However, we can now easily replace Descent with a more advanced optimiser such as ADAM."
},

{
    "location": "training/optimisers/#Flux.Optimise.Descent",
    "page": "Optimisers",
    "title": "Flux.Optimise.Descent",
    "category": "type",
    "text": "Descent(η)\n\nClassic gradient descent optimiser with learning rate η. For each parameter p and its gradient δp, this runs p -= η*δp.\n\n\n\n\n\n"
},

{
    "location": "training/optimisers/#Flux.Optimise.Momentum",
    "page": "Optimisers",
    "title": "Flux.Optimise.Momentum",
    "category": "type",
    "text": "Momentum(params, η = 0.01; ρ = 0.9)\n\nGradient descent with learning rate η and momentum ρ.\n\n\n\n\n\n"
},

{
    "location": "training/optimisers/#Flux.Optimise.Nesterov",
    "page": "Optimisers",
    "title": "Flux.Optimise.Nesterov",
    "category": "type",
    "text": "Nesterov(eta, ρ = 0.9)\n\nGradient descent with learning rate  η and Nesterov momentum ρ.\n\n\n\n\n\n"
},

{
    "location": "training/optimisers/#Flux.Optimise.ADAM",
    "page": "Optimisers",
    "title": "Flux.Optimise.ADAM",
    "category": "type",
    "text": "ADAM(η = 0.001, β = (0.9, 0.999))\n\nADAM optimiser.\n\n\n\n\n\n"
},

{
    "location": "training/optimisers/#Optimiser-Reference-1",
    "page": "Optimisers",
    "title": "Optimiser Reference",
    "category": "section",
    "text": "All optimisers return an object that, when passed to train!, will update the parameters passed to it.Descent\nMomentum\nNesterov\nADAM"
},

{
    "location": "training/training/#",
    "page": "Training",
    "title": "Training",
    "category": "page",
    "text": ""
},

{
    "location": "training/training/#Training-1",
    "page": "Training",
    "title": "Training",
    "category": "section",
    "text": "To actually train a model we need three things:A objective function, that evaluates how well a model is doing given some input data.\nA collection of data points that will be provided to the objective function.\nAn optimiser that will update the model parameters appropriately.With these we can call Flux.train!:Flux.train!(objective, params, data, opt)There are plenty of examples in the model zoo."
},

{
    "location": "training/training/#Loss-Functions-1",
    "page": "Training",
    "title": "Loss Functions",
    "category": "section",
    "text": "The objective function must return a number representing how far the model is from its target – the loss of the model. The loss function that we defined in basics will work as an objective. We can also define an objective in terms of some model:m = Chain(\n  Dense(784, 32, σ),\n  Dense(32, 10), softmax)\n\nloss(x, y) = Flux.mse(m(x), y)\nps = Flux.params(m)\n\n# later\nFlux.train!(loss, ps, data, opt)The objective will almost always be defined in terms of some cost function that measures the distance of the prediction m(x) from the target y. Flux has several of these built in, like mse for mean squared error or crossentropy for cross entropy loss, but you can calculate it however you want."
},

{
    "location": "training/training/#Datasets-1",
    "page": "Training",
    "title": "Datasets",
    "category": "section",
    "text": "The data argument provides a collection of data to train with (usually a set of inputs x and target outputs y). For example, here\'s a dummy data set with only one data point:x = rand(784)\ny = rand(10)\ndata = [(x, y)]Flux.train! will call loss(x, y), calculate gradients, update the weights and then move on to the next data point if there is one. We can train the model on the same data three times:data = [(x, y), (x, y), (x, y)]\n# Or equivalently\ndata = Iterators.repeated((x, y), 3)It\'s common to load the xs and ys separately. In this case you can use zip:xs = [rand(784), rand(784), rand(784)]\nys = [rand( 10), rand( 10), rand( 10)]\ndata = zip(xs, ys)Note that, by default, train! only loops over the data once (a single \"epoch\"). A convenient way to run multiple epochs from the REPL is provided by @epochs.julia> using Flux: @epochs\n\njulia> @epochs 2 println(\"hello\")\nINFO: Epoch 1\nhello\nINFO: Epoch 2\nhello\n\njulia> @epochs 2 Flux.train!(...)\n# Train for two epochs"
},

{
    "location": "training/training/#Callbacks-1",
    "page": "Training",
    "title": "Callbacks",
    "category": "section",
    "text": "train! takes an additional argument, cb, that\'s used for callbacks so that you can observe the training process. For example:train!(objective, ps, data, opt, cb = () -> println(\"training\"))Callbacks are called for every batch of training data. You can slow this down using Flux.throttle(f, timeout) which prevents f from being called more than once every timeout seconds.A more typical callback might look like this:test_x, test_y = # ... create single batch of test data ...\nevalcb() = @show(loss(test_x, test_y))\n\nFlux.train!(objective, ps, data, opt,\n            cb = throttle(evalcb, 5))Calling Flux.stop() in a callback will exit the training loop early.cb = function ()\n  accuracy() > 0.9 && Flux.stop()\nend"
},

{
    "location": "data/onehot/#",
    "page": "One-Hot Encoding",
    "title": "One-Hot Encoding",
    "category": "page",
    "text": ""
},

{
    "location": "data/onehot/#One-Hot-Encoding-1",
    "page": "One-Hot Encoding",
    "title": "One-Hot Encoding",
    "category": "section",
    "text": "It\'s common to encode categorical variables (like true, false or cat, dog) in \"one-of-k\" or \"one-hot\" form. Flux provides the onehot function to make this easy.julia> using Flux: onehot, onecold\n\njulia> onehot(:b, [:a, :b, :c])\n3-element Flux.OneHotVector:\n false\n  true\n false\n\njulia> onehot(:c, [:a, :b, :c])\n3-element Flux.OneHotVector:\n false\n false\n  trueThe inverse is onecold (which can take a general probability distribution, as well as just booleans).julia> onecold(ans, [:a, :b, :c])\n:c\n\njulia> onecold([true, false, false], [:a, :b, :c])\n:a\n\njulia> onecold([0.3, 0.2, 0.5], [:a, :b, :c])\n:c"
},

{
    "location": "data/onehot/#Batches-1",
    "page": "One-Hot Encoding",
    "title": "Batches",
    "category": "section",
    "text": "onehotbatch creates a batch (matrix) of one-hot vectors, and onecold treats matrices as batches.julia> using Flux: onehotbatch\n\njulia> onehotbatch([:b, :a, :b], [:a, :b, :c])\n3×3 Flux.OneHotMatrix:\n false   true  false\n  true  false   true\n false  false  false\n\njulia> onecold(ans, [:a, :b, :c])\n3-element Array{Symbol,1}:\n  :b\n  :a\n  :bNote that these operations returned OneHotVector and OneHotMatrix rather than Arrays. OneHotVectors behave like normal vectors but avoid any unnecessary cost compared to using an integer index directly. For example, multiplying a matrix with a one-hot vector simply slices out the relevant row of the matrix under the hood."
},

{
    "location": "gpu/#",
    "page": "GPU Support",
    "title": "GPU Support",
    "category": "page",
    "text": ""
},

{
    "location": "gpu/#GPU-Support-1",
    "page": "GPU Support",
    "title": "GPU Support",
    "category": "section",
    "text": ""
},

{
    "location": "gpu/#Installation-1",
    "page": "GPU Support",
    "title": "Installation",
    "category": "section",
    "text": "To get GPU support for NVIDIA graphics cards, you need to install CuArrays.jlSteps neededInstall NVIDIA toolkit\nInstall NVIDIA cuDNN library\nIn Julia\'s terminal run ]add CuArrays"
},

{
    "location": "gpu/#GPU-Usage-1",
    "page": "GPU Support",
    "title": "GPU Usage",
    "category": "section",
    "text": "Support for array operations on other hardware backends, like GPUs, is provided by external packages like CuArrays. Flux is agnostic to array types, so we simply need to move model weights and data to the GPU and Flux will handle it.For example, we can use CuArrays (with the cu converter) to run our basic example on an NVIDIA GPU.(Note that you need to have CUDA available to use CuArrays – please see the CuArrays.jl instructions for more details.)using CuArrays\n\nW = cu(rand(2, 5)) # a 2×5 CuArray\nb = cu(rand(2))\n\npredict(x) = W*x .+ b\nloss(x, y) = sum((predict(x) .- y).^2)\n\nx, y = cu(rand(5)), cu(rand(2)) # Dummy data\nloss(x, y) # ~ 3Note that we convert both the parameters (W, b) and the data set (x, y) to cuda arrays. Taking derivatives and training works exactly as before.If you define a structured model, like a Dense layer or Chain, you just need to convert the internal parameters. Flux provides mapleaves, which allows you to alter all parameters of a model at once.d = Dense(10, 5, σ)\nd = mapleaves(cu, d)\nd.W # Tracked CuArray\nd(cu(rand(10))) # CuArray output\n\nm = Chain(Dense(10, 5, σ), Dense(5, 2), softmax)\nm = mapleaves(cu, m)\nd(cu(rand(10)))As a convenience, Flux provides the gpu function to convert models and data to the GPU if one is available. By default, it\'ll do nothing, but loading CuArrays will cause it to move data to the GPU instead.julia> using Flux, CuArrays\n\njulia> m = Dense(10,5) |> gpu\nDense(10, 5)\n\njulia> x = rand(10) |> gpu\n10-element CuArray{Float32,1}:\n 0.800225\n ⋮\n 0.511655\n\njulia> m(x)\nTracked 5-element CuArray{Float32,1}:\n -0.30535\n ⋮\n -0.618002The analogue cpu is also available for moving models and data back off of the GPU.julia> x = rand(10) |> gpu\n10-element CuArray{Float32,1}:\n 0.235164\n ⋮\n 0.192538\n\njulia> x |> cpu\n10-element Array{Float32,1}:\n 0.235164\n ⋮\n 0.192538"
},

{
    "location": "saving/#",
    "page": "Saving & Loading",
    "title": "Saving & Loading",
    "category": "page",
    "text": ""
},

{
    "location": "saving/#Saving-and-Loading-Models-1",
    "page": "Saving & Loading",
    "title": "Saving and Loading Models",
    "category": "section",
    "text": "You may wish to save models so that they can be loaded and run in a later session. The easiest way to do this is via BSON.jl.Save a model:julia> using Flux\n\njulia> model = Chain(Dense(10,5,relu),Dense(5,2),softmax)\nChain(Dense(10, 5, NNlib.relu), Dense(5, 2), NNlib.softmax)\n\njulia> using BSON: @save\n\njulia> @save \"mymodel.bson\" modelLoad it again:julia> using Flux\n\njulia> using BSON: @load\n\njulia> @load \"mymodel.bson\" model\n\njulia> model\nChain(Dense(10, 5, NNlib.relu), Dense(5, 2), NNlib.softmax)Models are just normal Julia structs, so it\'s fine to use any Julia storage format for this purpose. BSON.jl is particularly well supported and most likely to be forwards compatible (that is, models saved now will load in future versions of Flux).note: Note\nIf a saved model\'s weights are stored on the GPU, the model will not load later on if there is no GPU support available. It\'s best to move your model to the CPU with cpu(model) before saving it."
},

{
    "location": "saving/#Saving-Model-Weights-1",
    "page": "Saving & Loading",
    "title": "Saving Model Weights",
    "category": "section",
    "text": "In some cases it may be useful to save only the model parameters themselves, and rebuild the model architecture in your code. You can use params(model) to get model parameters. You can also use data.(params) to remove tracking.julia> using Flux\n\njulia> model = Chain(Dense(10,5,relu),Dense(5,2),softmax)\nChain(Dense(10, 5, NNlib.relu), Dense(5, 2), NNlib.softmax)\n\njulia> weights = Tracker.data.(params(model));\n\njulia> using BSON: @save\n\njulia> @save \"mymodel.bson\" weightsYou can easily load parameters back into a model with Flux.loadparams!.julia> using Flux\n\njulia> model = Chain(Dense(10,5,relu),Dense(5,2),softmax)\nChain(Dense(10, 5, NNlib.relu), Dense(5, 2), NNlib.softmax)\n\njulia> using BSON: @load\n\njulia> @load \"mymodel.bson\" weights\n\njulia> Flux.loadparams!(model, weights)The new model we created will now be identical to the one we saved parameters for."
},

{
    "location": "saving/#Checkpointing-1",
    "page": "Saving & Loading",
    "title": "Checkpointing",
    "category": "section",
    "text": "In longer training runs it\'s a good idea to periodically save your model, so that you can resume if training is interrupted (for example, if there\'s a power cut). You can do this by saving the model in the callback provided to train!.using Flux: throttle\nusing BSON: @save\n\nm = Chain(Dense(10,5,relu),Dense(5,2),softmax)\n\nevalcb = throttle(30) do\n  # Show loss\n  @save \"model-checkpoint.bson\" model\nendThis will update the \"model-checkpoint.bson\" file every thirty seconds.You can get more advanced by saving a series of models throughout training, for example@save \"model-$(now()).bson\" modelwill produce a series of models like \"model-2018-03-06T02:57:10.41.bson\". You could also store the current test set loss, so that it\'s easy to (for example) revert to an older copy of the model if it starts to overfit.@save \"model-$(now()).bson\" model loss = testloss()You can even store optimiser state alongside the model, to resume training exactly where you left off.opt = ADAM(params(model))\n@save \"model-$(now()).bson\" model opt"
},

{
    "location": "performance/#",
    "page": "Performance Tips",
    "title": "Performance Tips",
    "category": "page",
    "text": ""
},

{
    "location": "performance/#Performance-Tips-1",
    "page": "Performance Tips",
    "title": "Performance Tips",
    "category": "section",
    "text": "All the usual Julia performance tips apply. As always profiling your code is generally a useful way of finding bottlenecks. Below follow some Flux specific tips/reminders."
},

{
    "location": "performance/#Don\'t-use-more-precision-than-you-need.-1",
    "page": "Performance Tips",
    "title": "Don\'t use more precision than you need.",
    "category": "section",
    "text": "Flux works great with all kinds of number types. But often you do not need to be working with say Float64 (let alone BigFloat). Switching to Float32 can give you a significant speed up, not because the operations are faster, but because the memory usage is halved. Which means allocations occur much faster. And you use less memory."
},

{
    "location": "performance/#Make-sure-your-custom-activation-functions-preserve-the-type-of-their-inputs-1",
    "page": "Performance Tips",
    "title": "Make sure your custom activation functions preserve the type of their inputs",
    "category": "section",
    "text": "Not only should your activation functions be type-stable, they should also preserve the type of their inputs.A very artificial example using an activatioon function like    my_tanh(x) = Float64(tanh(x))will result in performance on Float32 input orders of magnitude slower than the normal tanh would, because it results in having to use slow mixed type multiplication in the dense layers.Which means if you change your data say from Float64 to Float32 (which should give a speedup: see above), you will see a large slow-downThis can occur sneakily, because you can cause type-promotion by interacting with a numeric literals. E.g. the following will have run into the same problem as above:    leaky_tanh(x) = 0.01x + tanh(x)While one could change your activation function (e.g. to use 0.01f0x) to avoid this when ever your inputs change, the idiomatic (and safe way) is to use oftype.    leaky_tanh(x) = oftype(x/1, 0.01) + tanh(x)"
},

{
    "location": "performance/#Evaluate-batches-as-Matrices-of-features,-rather-than-sequences-of-Vector-features-1",
    "page": "Performance Tips",
    "title": "Evaluate batches as Matrices of features, rather than sequences of Vector features",
    "category": "section",
    "text": "While it can sometimes be tempting to process your observations (feature vectors) one at a time e.g.function loss_total(xs::AbstractVector{<:Vector}, ys::AbstractVector{<:Vector})\n    sum(zip(xs, ys)) do (x, y_target)\n        y_pred = model(x) #  evaluate the model\n        return loss(y_pred, y_target)\n    end\nendIt is much faster to concatenate them into a matrix, as this will hit BLAS matrix-matrix multiplication, which is much faster than the equivalent sequence of matrix-vector multiplications. Even though this means allocating new memory to store them contiguously.x_batch = reduce(hcat, xs)\ny_batch = reduce(hcat, ys)\n...\nfunction loss_total(x_batch::Matrix, y_batch::Matrix)\n    y_preds = model(x_batch)\n    sum(loss.(y_preds, y_batch))\nendWhen doing this kind of concatenation use reduce(hcat, xs) rather than hcat(xs...). This will avoid the splatting penality, and will hit the optimised reduce method."
},

{
    "location": "internals/tracker/#",
    "page": "Backpropagation",
    "title": "Backpropagation",
    "category": "page",
    "text": ""
},

{
    "location": "internals/tracker/#Flux.Tracker-1",
    "page": "Backpropagation",
    "title": "Flux.Tracker",
    "category": "section",
    "text": "Backpropagation, or reverse-mode automatic differentiation, is handled by the Flux.Tracker module.julia> using Flux.TrackerHere we discuss some more advanced uses of this module, as well as covering its internals."
},

{
    "location": "internals/tracker/#Taking-Gradients-1",
    "page": "Backpropagation",
    "title": "Taking Gradients",
    "category": "section",
    "text": "In the basics section we covered basic usage of the gradient function.using Flux.Tracker\n\nTracker.gradient((a, b) -> a*b, 2, 3) # (3.0 (tracked), 2.0 (tracked))gradient is actually just a thin wrapper around the backpropagator-based interface, forward.using Flux.Tracker: forward\n\ny, back = forward((a, b) -> a*b, 2, 3) # (6.0 (tracked), Flux.Tracker.#9)\n\nback(1) # (3.0 (tracked), 2.0 (tracked))The forward function returns two results. The first, y, is the original value of the function (perhaps with tracking applied). The second, back, is a new function which, given a sensitivity, returns the sensitivity of the inputs to forward (we call this a \"backpropagator\"). One use of this interface is to provide custom sensitivities when outputs are not scalar.julia> y, back = forward((a, b) -> a.*b, [1,2,3],[4,5,6])\n(param([4.0, 10.0, 18.0]), Flux.Tracker.#9)\n\njulia> back([1,1,1])\n(param([4.0, 5.0, 6.0]), param([1.0, 2.0, 3.0]))We can also take gradients in-place. This can be useful if you only care about first-order gradients.a, b = param(2), param(3)\n\nc = a*b # 6.0 (tracked)\n\nTracker.back!(c)\n\nTracker.grad(a), Tracker.grad(b) # (3.0, 2.0)"
},

{
    "location": "internals/tracker/#Tracked-Arrays-1",
    "page": "Backpropagation",
    "title": "Tracked Arrays",
    "category": "section",
    "text": "The param function converts a normal Julia array into a new object that, while behaving like an array, tracks extra information that allows us to calculate derivatives. For example, say we multiply two parameters:julia> W = param([1 2; 3 4])\nTracked 2×2 Array{Float64,2}:\n 1.0  2.0\n 3.0  4.0\n\njulia> x = param([5, 6])\nTracked 2-element Array{Float64,1}:\n 5.0\n 6.0\n\njulia> y = W*x\nTracked 2-element Array{Float64,1}:\n 17.0\n 39.0The output y is also a TrackedArray object. We can now backpropagate sensitivities to W and x via the back! function, and see the gradients accumulated in the W and x tracked arrays:julia> Tracker.back!(y, [1, -1])\n\njulia> W.grad\n2×2 Array{Float64,2}:\n 5.0   6.0\n-5.0  -6.0\n\njulia> x.grad\n2-element Array{Float64,1}:\n -2.0\n -2.0You may sometimes want to drop derivative information and just get the plain value back. You can do this by calling Tracker.data(W)."
},

{
    "location": "internals/tracker/#Custom-Gradients-1",
    "page": "Backpropagation",
    "title": "Custom Gradients",
    "category": "section",
    "text": "We can hook in to the processes above to implement custom gradients for a function or kernel. For a toy example, imagine a custom implementation of minus:minus(a, b) = a - bFirstly, we must tell the tracker system to stop when it sees a call to minus, and record it. We can do this using dispatch:using Flux.Tracker: TrackedArray, track, @grad\n\nminus(a::TrackedArray, b::TrackedArray) = track(minus, a, b)track takes care of building a new Tracked object and recording the operation on the tape. We just need to provide a gradient definition.@grad function minus(a, b)\n  return minus(data(a), data(b)), Δ -> (Δ, -Δ)\nendThis is essentially just a way of overloading the forward function we saw above. We strip tracking from a and b so that we are calling the original definition of minus (otherwise, we\'d just try to track the call again and hit an infinite regress).Note that in the backpropagator we don\'t call data(a); we do in fact want to track this, since nest AD will take a derivative through the backpropagator itself. For example, the gradient of * might look like this.@grad a * b = data(a)*data(b), Δ -> (Δ*b, a*Δ)We can then calculate the first derivative of minus as follows:a = param([1,2,3])\nb = param([3,2,1])\n\nc = minus(a, b)  # [-2.0 (tracked), 0.0 (tracked), 2.0 (tracked)]\n\nTracker.back!(c, 1)\nTracker.grad(a)  # [1.00, 1.00, 1.00]\nTracker.grad(b)  # [-1.00, -1.00, -1.00]For multi-argument functions with custom gradients, you likely want to catch not just minus(::TrackedArray, ::TrackedArray) but also minus(::Array, TrackedArray) and so on. To do so, just define those extra signatures as needed:minus(a::AbstractArray, b::TrackedArray) = Tracker.track(minus, a, b)\nminus(a::TrackedArray, b::AbstractArray) = Tracker.track(minus, a, b)"
},

{
    "location": "internals/tracker/#Tracked-Internals-1",
    "page": "Backpropagation",
    "title": "Tracked Internals",
    "category": "section",
    "text": "All Tracked* objects (TrackedArray, TrackedReal) are light wrappers around the Tracked type, which you can access via the .tracker field.julia> x.tracker\nFlux.Tracker.Tracked{Array{Float64,1}}(0x00000000, Flux.Tracker.Call{Nothing,Tuple{}}(nothing, ()), true, [5.0, 6.0], [-2.0, -2.0])The Tracker stores the gradient of a given object, which we\'ve seen before.julia> x.tracker.grad\n2-element Array{Float64,1}:\n -2.0\n -2.0The tracker also contains a Call object, which simply represents a function call that was made at some point during the forward pass. For example, the + call would look like this:julia> Tracker.Call(+, 1, 2)\nFlux.Tracker.Call{Base.#+,Tuple{Int64,Int64}}(+, (1, 2))In the case of the y we produced above, we can see that it stores the call that produced it – that is, W*x.julia> y.tracker.f\nFlux.Tracker.Call{...}(*, (param([1.0 2.0; 3.0 4.0]), param([5.0, 6.0])))Notice that because the arguments to the call may also be tracked arrays, storing their own calls, this means that Tracker ends up forming a data structure that records everything that happened during the forward pass (often known as a tape).When we call back!(y, [1, -1]), the sensitivities [1, -1] simply get forwarded to y\'s call (*), effectively callingTracker.back(*, [1, -1], W, x)which in turn calculates the sensitivities of the arguments (W and x) and back-propagates through their calls. This is recursive, so it will walk the entire program graph and propagate gradients to the original model parameters."
},

{
    "location": "community/#",
    "page": "Community",
    "title": "Community",
    "category": "page",
    "text": ""
},

{
    "location": "community/#Community-1",
    "page": "Community",
    "title": "Community",
    "category": "section",
    "text": "All Flux users are welcome to join our community on the Julia forum, the slack (channel #machine-learning), or Flux\'s Gitter. If you have questions or issues we\'ll try to help you out.If you\'re interested in hacking on Flux, the source code is open and easy to understand – it\'s all just the same Julia code you work with normally. You might be interested in our intro issues to get started."
},

]}
