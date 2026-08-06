var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Flux-1",
    "page": "Home",
    "title": "Flux",
    "category": "section",
    "text": "... Initialising Photon Beams ...Flux is a library for machine learning, implemented in Julia. In a nutshell, it simply lets you run normal Julia code on a backend like TensorFlow. It also provides many conveniences for doing deep learning.Flux is very flexible. You can use a convenient Keras-like API if you want something simple, but you can also drop down to straight mathematics, or build your own abstractions. You can even use Flux's utilities (like optimisers) with a completely different backend (like Knet) or mix and match approaches.Note that Flux is in alpha. Many things work but the API is still in a state of... well, it might change.Note: If you're using Julia v0.5 please see this version of the docs instead."
},

{
    "location": "index.html#Where-do-I-start?-1",
    "page": "Home",
    "title": "Where do I start?",
    "category": "section",
    "text": "... Charging Ion Capacitors ...The examples give a feel for high-level usage.If you want to know why Flux is unique, or just don't want to see those digits again, check out the model building guide instead.Flux is meant to be played with. These docs have lots of code snippets; try them out in  Juno!"
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "... Inflating Graviton Zeppelins ...Pkg.update()\nPkg.add(\"Flux.jl\")You'll also need a backend to run real training, if you don't have one already. Choose from MXNet or TensorFlow (MXNet is the recommended option if you're not sure):Pkg.add(\"MXNet\") # or \"TensorFlow\"\nPkg.test(\"Flux\") # Make sure everything installed properlyNote: TensorFlow integration may not work properly on Julia v0.6 yet."
},

{
    "location": "models/basics.html#",
    "page": "Model Building Basics",
    "title": "Model Building Basics",
    "category": "page",
    "text": ""
},

{
    "location": "models/basics.html#Model-Building-Basics-1",
    "page": "Model Building Basics",
    "title": "Model Building Basics",
    "category": "section",
    "text": ""
},

{
    "location": "models/basics.html#Net-Functions-1",
    "page": "Model Building Basics",
    "title": "Net Functions",
    "category": "section",
    "text": "Flux's core feature is the @net macro, which adds some superpowers to regular ol' Julia functions. Consider this simple function with the @net annotation applied:@net f(x) = x .* x\nf([1,2,3]) == [1,4,9]This behaves as expected, but we have some extra features. For example, we can convert the function to run on TensorFlow or MXNet:f_mxnet = mxnet(f)\nf_mxnet([1,2,3]) == [1.0, 4.0, 9.0]Simples! Flux took care of a lot of boilerplate for us and just ran the multiplication on MXNet. MXNet can optimise this code for us, taking advantage of parallelism or running the code on a GPU.Using MXNet, we can get the gradient of the function, too:back!(f_mxnet, [1,1,1], [1,2,3]) == ([2.0, 4.0, 6.0],)f is effectively x^2, so the gradient is 2x as expected."
},

{
    "location": "models/basics.html#The-Model-1",
    "page": "Model Building Basics",
    "title": "The Model",
    "category": "section",
    "text": "The core concept in Flux is the model. This corresponds to what might be called a \"layer\" or \"module\" in other frameworks. A model is simply a differentiable function with parameters. Given a model m we can do things like:m(x)           # See what the model does to an input vector `x`\nback!(m, Δ, x) # backpropogate the gradient `Δ` through `m`\nupdate!(m, η)  # update the parameters of `m` using the gradientWe can implement a model however we like as long as it fits this interface. But as hinted above, @net is a particularly easy way to do it, because it gives you these functions for free."
},

{
    "location": "models/basics.html#Parameters-1",
    "page": "Model Building Basics",
    "title": "Parameters",
    "category": "section",
    "text": "Consider how we'd write a logistic regression. We just take the Julia code and add @net.@net logistic(W, b, x) = softmax(x * W .+ b)\n\nW = randn(10, 2)\nb = randn(1, 2)\nx = rand(1, 10) # [0.563 0.346 0.780  …] – fake data\ny = [1 0] # our desired classification of `x`\n\nŷ = logistic(W, b, x) # [0.46 0.54]The network takes a set of 10 features (x, a row vector) and produces a classification ŷ, equivalent to a probability of true vs false. softmax scales the output to sum to one, so that we can interpret it as a probability distribution.We can use MXNet and get gradients:logisticm = mxnet(logistic)\nlogisticm(W, b, x) # [0.46 0.54]\nback!(logisticm, [0.1 -0.1], W, b, x) # (dW, db, dx)The gradient [0.1 -0.1] says that we want to increase ŷ[1] and decrease ŷ[2] to get closer to y. back! gives us the tweaks we need to make to each input (W, b, x) in order to do this. If we add these tweaks to W and b it will predict ŷ more accurately.Treating parameters like W and b as inputs can get unwieldy in larger networks. Since they are both global we can use them directly:@net logistic(x) = softmax(x * W .+ b)However, this gives us a problem: how do we get their gradients?Flux solves this with the Param wrapper:W = param(randn(10, 2))\nb = param(randn(1, 2))\n@net logistic(x) = softmax(x * W .+ b)This works as before, but now W.x stores the real value and W.Δx stores its gradient, so we don't have to manage it by hand. We can even use update! to apply the gradients automatically.logisticm(x) # [0.46, 0.54]\n\nback!(logisticm, [-1 1], x)\nupdate!(logisticm, 0.1)\n\nlogisticm(x) # [0.51, 0.49]Our network got a little closer to the target y. Now we just need to repeat this millions of times.Side note: We obviously need a way to calculate the \"tweak\" [0.1, -0.1] automatically. We can use a loss function like mean squared error for this:# How wrong is ŷ?\nmse([0.46, 0.54], [1, 0]) == 0.292\n# What change to `ŷ` will reduce the wrongness?\nback!(mse, -1, [0.46, 0.54], [1, 0]) == [0.54 -0.54]"
},

{
    "location": "models/basics.html#Layers-1",
    "page": "Model Building Basics",
    "title": "Layers",
    "category": "section",
    "text": "Bigger networks contain many affine transformations like W * x + b. We don't want to write out the definition every time we use it. Instead, we can factor this out by making a function that produces models:function create_affine(in, out)\n  W = param(randn(out,in))\n  b = param(randn(out))\n  @net x -> W * x + b\nend\n\naffine1 = create_affine(3,2)\naffine1([1,2,3])Flux has a more powerful syntax for this pattern, but also provides a bunch of layers out of the box. So we can instead write:affine1 = Affine(5, 5)\naffine2 = Affine(5, 5)\n\nsoftmax(affine1(x)) # [0.167952 0.186325 0.176683 0.238571 0.23047]\nsoftmax(affine2(x)) # [0.125361 0.246448 0.21966 0.124596 0.283935]"
},

{
    "location": "models/basics.html#Combining-Layers-1",
    "page": "Model Building Basics",
    "title": "Combining Layers",
    "category": "section",
    "text": "A more complex model usually involves many basic layers like affine, where we use the output of one layer as the input to the next:mymodel1(x) = softmax(affine2(σ(affine1(x))))\nmymodel1(x1) # [0.187935, 0.232237, 0.169824, 0.230589, 0.179414]This syntax is again a little unwieldy for larger networks, so Flux provides another template of sorts to create the function for us:mymodel2 = Chain(affine1, σ, affine2, softmax)\nmymodel2(x2) # [0.187935, 0.232237, 0.169824, 0.230589, 0.179414]mymodel2 is exactly equivalent to mymodel1 because it simply calls the provided functions in sequence. We don't have to predefine the affine layers and can also write this as:mymodel3 = Chain(\n  Affine(5, 5), σ,\n  Affine(5, 5), softmax)"
},

{
    "location": "models/basics.html#Dressed-like-a-model-1",
    "page": "Model Building Basics",
    "title": "Dressed like a model",
    "category": "section",
    "text": "We noted above that a model is a function with trainable parameters. Normal functions like exp are actually models too – they just happen to have 0 parameters. Flux doesn't care, and anywhere that you use one, you can use the other. For example, Chain will happily work with regular functions:foo = Chain(exp, sum, log)\nfoo([1,2,3]) == 3.408 == log(sum(exp([1,2,3])))"
},

{
    "location": "models/templates.html#",
    "page": "Model Templates",
    "title": "Model Templates",
    "category": "page",
    "text": ""
},

{
    "location": "models/templates.html#Model-Templates-1",
    "page": "Model Templates",
    "title": "Model Templates",
    "category": "section",
    "text": "We mentioned that we could factor out the repetition of defining affine layers with something like:function create_affine(in, out)\n  W = param(randn(out,in))\n  b = param(randn(out))\n  @net x -> W * x + b\nend@net type syntax provides a shortcut for this:@net type MyAffine\n  W\n  b\n  x -> x * W + b\nend\n\n# Convenience constructor\nMyAffine(in::Integer, out::Integer) =\n  MyAffine(randn(out, in), randn(out))\n\nmodel = Chain(MyAffine(5, 5), MyAffine(5, 5))\n\nmodel(x1) # [-1.54458,0.492025,0.88687,1.93834,-4.70062]This is almost exactly how Affine is defined in Flux itself. Using @net type gives us some extra conveniences:It creates default constructor MyAffine(::AbstractArray, ::AbstractArray) which initialises params for us;\nIt subtypes Flux.Model to explicitly mark this as a model;\nWe can easily define custom constructors or instantiate Affine with arbitrary weights of our choosing;\nWe can dispatch on the Affine type, for example to override how it gets converted to MXNet, or to hook into shape inference."
},

{
    "location": "models/templates.html#Models-in-templates-1",
    "page": "Model Templates",
    "title": "Models in templates",
    "category": "section",
    "text": "@net models can contain sub-models as well as just array parameters:@net type TLP\n  first\n  second\n  function (x)\n    l1 = σ(first(x))\n    l2 = softmax(second(l1))\n  end\nendClearly, the first and second parameters are not arrays here, but should be models themselves, and produce a result when called with an input array x. The Affine layer fits the bill, so we can instantiate TLP with two of them:model = TLP(Affine(10, 20),\n            Affine(20, 15))\nx1 = rand(20)\nmodel(x1) # [0.057852,0.0409741,0.0609625,0.0575354 ...You may recognise this as being equivalent toChain(\n  Affine(10, 20), σ\n  Affine(20, 15), softmax)"
},

{
    "location": "models/templates.html#Supported-syntax-1",
    "page": "Model Templates",
    "title": "Supported syntax",
    "category": "section",
    "text": "The syntax used to define a forward pass like x -> x*W + b behaves exactly like Julia code for the most part. However, it's important to remember that it's defining a dataflow graph, not a general Julia expression. In practice this means that anything side-effectful, or things like control flow and printlns, won't work as expected. In future we'll continue to expand support for Julia syntax and features."
},

{
    "location": "models/recurrent.html#",
    "page": "Recurrence",
    "title": "Recurrence",
    "category": "page",
    "text": ""
},

{
    "location": "models/recurrent.html#Recurrent-Models-1",
    "page": "Recurrence",
    "title": "Recurrent Models",
    "category": "section",
    "text": "Recurrence is a first-class feature in Flux and recurrent models are very easy to build and use. Recurrences are often illustrated as cycles or self-dependencies in the graph; they can also be thought of as a hidden output from / input to the network. For example, for a sequence of inputs x1, x2, x3 ... we produce predictions as follows:y1 = f(W, x1) # `f` is the model, `W` represents the parameters\ny2 = f(W, x2)\ny3 = f(W, x3)\n...Each evaluation is independent and the prediction made for a given input will always be the same. That makes a lot of sense for, say, MNIST images, but less sense when predicting a sequence. For that case we introduce the hidden state:y1, s = f(W, x1, s)\ny2, s = f(W, x2, s)\ny3, s = f(W, x3, s)\n...The state s allows the prediction to depend not only on the current input x but also on the history of past inputs.The simplest recurrent network looks as follows in Flux, and it should be familiar if you've seen the equations defining an RNN before:@net type Recurrent\n  Wxy; Wyy; by\n  y\n  function (x)\n    y = tanh( x * Wxy + y{-1} * Wyy + by )\n  end\nendThe only difference from a regular feed-forward layer is that we create a variable y which is defined as depending on itself. The y{-1} syntax means \"take the value of y from the previous run of the network\".Using recurrent layers is straightforward and no different feedforward ones in terms of the Chain macro etc. For example:model = Chain(\n    Affine(784, 20), σ\n    Recurrent(20, 30),\n    Recurrent(30, 15))Before using the model we need to unroll it. This happens with the unroll function:unroll(model, 20)This call creates an unrolled, feed-forward version of the model which accepts N (= 20) inputs and generates N predictions at a time. Essentially, the model is replicated N times and Flux ties the hidden outputs y to hidden inputs.Here's a more complex recurrent layer, an LSTM, and again it should be familiar if you've seen the equations:@net type LSTM\n  Wxf; Wyf; bf\n  Wxi; Wyi; bi\n  Wxo; Wyo; bo\n  Wxc; Wyc; bc\n  y; state\n  function (x)\n    # Gates\n    forget = σ( x * Wxf + y{-1} * Wyf + bf )\n    input  = σ( x * Wxi + y{-1} * Wyi + bi )\n    output = σ( x * Wxo + y{-1} * Wyo + bo )\n    # State update and output\n    state′ = tanh( x * Wxc + y{-1} * Wyc + bc )\n    state  = forget .* state{-1} + input .* state′\n    y = output .* tanh(state)\n  end\nendThe only unfamiliar part is that we have to define all of the parameters of the LSTM upfront, which adds a few lines at the beginning."
},

{
    "location": "models/debugging.html#",
    "page": "Debugging",
    "title": "Debugging",
    "category": "page",
    "text": ""
},

{
    "location": "models/debugging.html#Debugging-Models-1",
    "page": "Debugging",
    "title": "Debugging Models",
    "category": "section",
    "text": "Let's take our two-layer perceptron as an example again, running on MXNet:@net type TLP\n  first\n  second\n  function (x)\n    l1 = σ(first(x))\n    l2 = softmax(second(l1))\n  end\nend\n\nmodel = TLP(Affine(10, 20), Affine(21, 15))\n\nmxmodel = mxnet(model)\n\nmxmodel(rand(10))Unfortunately, this model has a (fairly obvious) typo, which means that the code above won't run. Instead we get an error message:Error in operator dot2: [21:28:21] src/operator/tensor/./matrix_op-inl.h:460:\nCheck failed: lshape[1] == rshape[0] (20 vs. 21) dot shape error: (1,20) X (21,15)\nFlux.Affine at affine.jl:8\nTLP at basic.jl:6\n(::Flux.MX.Model)(::Flux.Batch{Array{Float64,1},Array{Float64,2}}) at model.jl:105\n(::Flux.MX.Model)(::Array{Float64,1}) at model.jl:107Most frameworks would only give the error message here – not so helpful if you have thousands of nodes in your computational graph. However, Flux is able to give good error reports even when no Julia code has been run, e.g. when running on a backend like MXNet. This enables us to pinpoint the source of the error very quickly even in a large model.In this case, we can immediately see that the error occurred within an Affine layer. There are two such layers, but this one was called from the second line of TLP, so it must be the second Affine layer we defined. The layer expected an input of length 21 but got 20 instead.Of course, often a stack trace isn't enough to figure out the source of an error. Another option is to simply step through the execution of the model using Gallium. While handy, however, stepping isn't always the best way to get a \"bird's eye view\" of the code. For that, Flux provides a macro called @shapes:julia> @shapes model(rand(5,10))\n\n# /Users/mike/test.jl, line 18:\ngull = σ(Affine(10, 20)(Input()[1]::(5,10))::(5,20))::(5,20)\n# /Users/mike/.julia/v0.6/Flux/src/layers/affine.jl, line 8:\nlobster = gull * _::(21,15) + _::(1,15)\n# /Users/mike/test.jl, line 19:\nraven = softmax(lobster)This is a lot like Julia's own code_warntype; but instead of annotating expressions with types, we display their shapes. As a lowered form it has some quirks; input arguments are represented by Input()[N] and parameters by an underscore.This makes the problem fairly obvious. We tried to multiply the output of the first layer (5, 20) by a parameter (21, 15); the inner dimensions should have been equal.Notice that while the first Affine layer is displayed as-is, the second was inlined and we see a reference to where the W * x + b line was defined in Flux's source code. In this way Flux makes it easy to drill down into problem areas, without showing you the full graph of thousands of nodes at once.With the typo fixed, the output of @shapes looks as follows:# /Users/mike/test.jl, line 18:\nopossum = σ(Affine(10, 20)(Input()[1]::(5,10))::(5,20))::(5,20)\n# /Users/mike/test.jl, line 19:\nwren = softmax(Affine(20, 15)(opossum)::(5,15))::(5,15)"
},

{
    "location": "apis/batching.html#",
    "page": "Batching",
    "title": "Batching",
    "category": "page",
    "text": ""
},

{
    "location": "apis/batching.html#Batching-1",
    "page": "Batching",
    "title": "Batching",
    "category": "section",
    "text": ""
},

{
    "location": "apis/batching.html#Basics-1",
    "page": "Batching",
    "title": "Basics",
    "category": "section",
    "text": "Existing machine learning frameworks and libraries represent batching, and other properties of data, only implicitly. Your machine learning data is a large N-dimensional array, which may have a shape like:100 × 50 × 256 × 256Typically, this might represent that you have (say) a batch of 100 samples, where each sample is a 50-long sequence of 256×256 images. This is great for performance, but array operations often become much more cumbersome as a result. Especially if you manipulate dimensions at runtime as an optimisation, debugging models can become extremely fiddly, with a proliferation of X × Y × Z arrays and no information about where they came from.Flux introduces a new approach where the batch dimension is represented explicitly as part of the data. For example:julia> xs = Batch([[1,2,3], [4,5,6]])\n2-element Batch of Vector{Int64}:\n [1,2,3]\n [4,5,6]Batches are represented the way we think about them; as a list of data points. We can do all the usual array operations with them, including getting the first with xs[1], iterating over them and so on. The trick is that under the hood, the data is batched into a single array:julia> rawbatch(xs)\n2×3 Array{Int64,2}:\n 1  2  3\n 4  5  6When we put a Batch object into a model, the model is ultimately working with a single array, which means there's no performance overhead and we get the full benefit of standard batching.Turning a set of vectors into a matrix is fairly easy anyway, so what's the big deal? Well, it gets more interesting as we start working with more complex data. Say we were working with 4×4 images:julia> xs = Batch([[1 2; 3 4], [5 6; 7 8]])\n2-element Flux.Batch of Array{Int64,2}:\n [1 2; 3 4]\n [5 6; 7 8]The raw batch array is much messier, and harder to recognise:julia> rawbatch(xs)\n2×2×2 Array{Int64,3}:\n[:, :, 1] =\n 1  3\n 5  7\n\n[:, :, 2] =\n 2  4\n 6  8Furthermore, because the batches acts like a list of arrays, we can use simple and familiar operations on it:julia> map(flatten, xs)\n2-element Array{Array{Int64,1},1}:\n [1,3,2,4]\n [5,7,6,8]flatten is simple enough over a single data point, but flattening a batched data set is more complex and you end up needing arcane array operations like mapslices. A Batch can just handle this for you for free, and more importantly it ensures that your operations are correct – that you haven't mixed up your batch and data dimensions, or used the wrong array op, and so on."
},

{
    "location": "apis/batching.html#Sequences-and-Nesting-1",
    "page": "Batching",
    "title": "Sequences and Nesting",
    "category": "section",
    "text": "As well as Batch, there's a structure called Seq which behaves very similarly. Let's say we have two one-hot encoded DNA sequences:julia> x1 = Seq([[0,1,0,0], [1,0,0,0], [0,0,0,1]]) # [A, T, C, G]\njulia> x2 = Seq([[0,0,1,0], [0,0,0,1], [0,0,1,0]])\n\njulia> rawbatch(x1)\n3×4 Array{Int64,2}:\n 0  1  0  0\n 1  0  0  0\n 0  0  0  1This is identical to Batch so far; but where it gets interesting is that you can actually nest these types:julia> xs = Batch([x1, x2])\n2-element Batch of Seq of Vector{Int64}:\n [[0,1,0,0],[1,0,0,0],[0,0,0,1]]\n [[0,0,1,0],[0,0,0,1],[0,0,1,0]]Again, this represents itself intuitively as a list-of-lists-of-lists, but rawbatch shows that the real underlying value is an Array{Int64,3} of shape 2×3×4."
},

{
    "location": "apis/batching.html#Future-Work-1",
    "page": "Batching",
    "title": "Future Work",
    "category": "section",
    "text": "The design of batching is still a fairly early work in progress, though it's used in a few places in the system. For example, all Flux models expect to be given Batch objects which are unwrapped into raw arrays for the computation. Models will convert their arguments if necessary, so it's convenient to call a model with a single data point like f([1,2,3]).Right now, the Batch or Seq types always stack along the left-most dimension. In future, this will be customisable, and Flux will provide implementations of common functions that are generic across the batch dimension. This brings the following benefits:Code can be written in a batch-agnostic way or be generic across batching strategies.\nBatching and optimisations, like switching batch dimensions, can be expressed by the programmer with compiler support; fewer code changes are required and optimisations are guaranteed not to break the model.\nThis also opens the door for more automatic optimisations, e.g. having the compiler explore the search base of possible batching combinations.Here's a more detailed illustration of how it might look for code to be \"generic across batching\". Take for example a weight matrix W times a vector x, as used in a logistic regression or a simple neural network:   W    *   x  =>   y\n(10×28) * (28) => (10)If we want to work with a batch of 50 xs, one option is to stack the data into a matrix of size 28 × 50.   W    *    x    =>    y\n(10×28) * (28×50) => (10×50)This works, but we may find that it's slow or doesn't fit well with the rest of the model, which batches on the first dimension. For that reason we may instead want to put the data in a 50 × 28 matrix and alter the code as follows:   x    *    W'   =>    y\n(50×28) * (28×10) => (50×10)to make the shapes work out. This code change is not ideal; in more complex cases it can become fiddly and error-prone, and it means that the code is less reusable, tied to a particular implementation strategy.There's an alternative. We keep the same code, but represent the batched xs as either a Batch{Vector,1} or a Batch{Vector,2}, depending on how the data is stacked. Then we can simply overload * as follows:*(W::Matrix, x::Batch{Vector,1}) = x * W'\n*(W::Matrix, x::Batch{Vector,2}) = W * xThis means that we can always write W*x, and the code is reusable in a larger network regardless of the overall batching approach. Moreover, Julia's type system ensures there's no runtime cost to doing this, and we can compile the code appropriately for backends like TensorFlow as well."
},

{
    "location": "apis/backends.html#",
    "page": "Backends",
    "title": "Backends",
    "category": "page",
    "text": ""
},

{
    "location": "apis/backends.html#Backends-1",
    "page": "Backends",
    "title": "Backends",
    "category": "section",
    "text": ""
},

{
    "location": "apis/backends.html#Basic-Usage-1",
    "page": "Backends",
    "title": "Basic Usage",
    "category": "section",
    "text": "model = Chain(Affine(10, 20), σ, Affine(20, 15), softmax)\nxs = rand(10)Currently, Flux's pure-Julia backend has no optimisations. This means that callingmodel(rand(10)) #> [0.0650, 0.0655, ...]directly won't have great performance. In order to run a computationally intensive training process, we need to use a backend like MXNet or TensorFlow.This is easy to do. Just call either mxnet or tf on a model to convert it to a model of that kind:mxmodel = mxnet(model)\nmxmodel(xs) #> [0.0650, 0.0655, ...]\n# or\ntfmodel = tf(model)\ntfmodel(xs) #> [0.0650, 0.0655, ...]These new models look and feel exactly like every other model in Flux, including returning the same result when you call them, and can be trained as usual using Flux.train!(). The difference is that the computation is being carried out by a backend, which will usually give a large speedup."
},

{
    "location": "apis/backends.html#Native-Integration-1",
    "page": "Backends",
    "title": "Native Integration",
    "category": "section",
    "text": "Flux aims to provide high-level APIs that work well across backends, but in some cases you may want to take advantage of features specific to a given backend. In these cases it's easy to \"drop down\" and use the backend's API directly, where appropriate. For example:using MXNet\nFlux.loadmx()\n\nmxmodel = mx.FeedForward(model)This returns a standard mx.FeedForward instance, just like you might have created using MXNet's usual API. You can then use this with MXNet's data provider implementation, custom optimisers, or distributed training processes.Same goes for TensorFlow, where it's easy to create a Tensor object:using TensorFlow\nFlux.loadtf()\n\nx  = placeholder(Float32)\ny = Tensor(model, x)This makes makes it easy to take advantage of Flux's model description and debugging tools while also getting the benefit of the work put into these backends. You can check out how this looks with the integration examples here."
},

{
    "location": "apis/storage.html#",
    "page": "Storing Models",
    "title": "Storing Models",
    "category": "page",
    "text": ""
},

{
    "location": "apis/storage.html#Loading-and-Saving-Models-1",
    "page": "Storing Models",
    "title": "Loading and Saving Models",
    "category": "section",
    "text": "model = Chain(Affine(10, 20), σ, Affine(20, 15), softmax)Since models are just simple Julia data structures, it's very easy to save and load them using any of Julia's existing serialisation formats. For example, using Julia's built-in serialize:open(io -> serialize(io, model), \"model.jls\", \"w\")\nopen(io -> deserialize(io), \"model.jls\")One issue with serialize is that it doesn't promise compatibility between major Julia versions. For longer-term storage it's good to use a package like JLD.using JLD\n@save \"model.jld\" model\n@load \"model.jld\"However, JLD will break for some models as functions are not supported on 0.5+. You can resolve that by checking out this branch.Right now this is the only storage format Flux supports. In future Flux will support loading and saving other model formats (on an as-needed basis)."
},

{
    "location": "examples/logreg.html#",
    "page": "Simple MNIST",
    "title": "Simple MNIST",
    "category": "page",
    "text": ""
},

{
    "location": "examples/logreg.html#Recognising-MNIST-Digits-1",
    "page": "Simple MNIST",
    "title": "Recognising MNIST Digits",
    "category": "section",
    "text": "This walkthrough example will take you through writing a multi-layer perceptron that classifies MNIST digits with high accuracy.First, we load the data using the MNIST package:using Flux, MNIST\nusing Flux: accuracy\n\ndata = [(trainfeatures(i), onehot(trainlabel(i), 0:9)) for i = 1:60_000]\ntrain = data[1:50_000]\ntest = data[50_001:60_000]The only Flux-specific function here is onehot, which takes a class label and turns it into a one-hot-encoded vector that we can use for training. For example:julia> onehot(:b, [:a, :b, :c])\n3-element Array{Int64,1}:\n 0\n 1\n 0Otherwise, the format of the data is simple enough, it's just a list of tuples from input to output. For example:julia> data[1]\n([0.0,0.0,0.0, … 0.0,0.0,0.0],[0,0,0,0,0,1,0,0,0,0])data[1][1] is a 28*28 == 784 length vector (mostly zeros due to the black background) and data[1][2] is its classification.Now we define our model, which will simply be a function from one to the other.m = @Chain(\n  Input(784),\n  Affine(128), relu,\n  Affine( 64), relu,\n  Affine( 10), softmax)\n\nmodel = mxnet(m) # Convert to MXNetWe can try this out on our data already:julia> model(tobatch(data[1][1]))\n10-element Array{Float64,1}:\n 0.10614  \n 0.0850447\n 0.101474\n ...The model gives a probability of about 0.1 to each class – which is a way of saying, \"I have no idea\". This isn't too surprising as we haven't shown it any data yet. This is easy to fix:Flux.train!(model, train, η = 1e-3,\n            cb = [()->@show accuracy(m, test)])The training step takes about 5 minutes (to make it faster we can do smarter things like batching). If you run this code in Juno, you'll see a progress meter, which you can hover over to see the remaining computation time.Towards the end of the training process, Flux will have reported that the accuracy of the model is now about 90%. We can try it on our data again:10-element Array{Float32,1}:\n ...\n 5.11423f-7\n 0.9354     \n 3.1033f-5  \n 0.000127077\n ...Notice the class at 93%, suggesting our model is very confident about this image. We can use onecold to compare the true and predicted classes:julia> onecold(data[1][2], 0:9)\n5\n\njulia> onecold(model(tobatch(data[1][1])), 0:9)\n5Success!"
},

{
    "location": "examples/char-rnn.html#",
    "page": "Char RNN",
    "title": "Char RNN",
    "category": "page",
    "text": ""
},

{
    "location": "examples/char-rnn.html#Char-RNN-1",
    "page": "Char RNN",
    "title": "Char RNN",
    "category": "section",
    "text": "This walkthrough will take you through a model like that used in Karpathy's 2015 blog post, which can learn to generate text in the style of Shakespeare (or whatever else you may use as input). shakespeare_input.txt is here.using Flux\nimport StatsBase: wsampleFirstly, we define up front how many steps we want to unroll the RNN, and the number of data points to batch together. Then we create some functions to prepare our data, using Flux's built-in utilities.nunroll = 50\nnbatch = 50\n\ngetseqs(chars, alphabet) =\n  sequences((onehot(Float32, char, alphabet) for char in chars), nunroll)\ngetbatches(chars, alphabet) =\n  batches((getseqs(part, alphabet) for part in chunk(chars, nbatch))...)Because we want the RNN to predict the next letter at each iteration, our target data is simply our input data offset by one. For example, if the input is \"The quick brown fox\", the target will be \"he quick brown fox \". Each letter is one-hot encoded and sequences are batched together to create the training data.input = readstring(\"shakespeare_input.txt\");\nalphabet = unique(input)\nN = length(alphabet)\n\n# An iterator of (input, output) pairs\ntrain = zip(getbatches(input, alphabet), getbatches(input[2:end], alphabet))\n# We will evaluate the loss on a particular batch to monitor the training.\neval = tobatch.(first(drop(train, 5)))Creating the model and training it is straightforward:model = Chain(\n  Input(N),\n  LSTM(N, 256),\n  LSTM(256, 256),\n  Affine(256, N),\n  softmax)\n\nm = tf(unroll(model, nunroll))\n\n# Call this to see how the model is doing\nevalcb = () -> @show logloss(m(eval[1]), eval[2])\n\n@time Flux.train!(m, train, η = 0.1, loss = logloss, cb = [evalcb])\nFinally, we can sample the model. For sampling we remove the softmax from the end of the chain so that we can \"sharpen\" the resulting probabilities.function sample(model, n, temp = 1)\n  s = [rand(alphabet)]\n  m = unroll1(model)\n  for i = 1:n-1\n    push!(s, wsample(alphabet, softmax(m(unsqueeze(onehot(s[end], alphabet)))./temp)[1,:]))\n  end\n  return string(s...)\nend\n\nsample(model[1:end-1], 100)sample then produces a string of Shakespeare-like text. This won't produce great results after only a single epoch (though they will be recognisably different from the untrained model). Going for 30 epochs or so produces good results.Trained on a dataset from base Julia, the network can produce code like:function show(io::IO, md::Githompty)\n    Buffer(jowerTriangular(inals[i], initabs_indices), characters, side, nextfloat(typeof(x)))\n    isnull(r) && return\n    start::I!\n    for j = 1:length(b,1)\n        a = s->cosvect(code)\n        return\n    end\n    indsERenv | maximum(func,lsg))\n    for i = 1:last(Abjelar) && fname (=== nothing)\n        throw(ArgumentError(\"read is declave non-fast-a/remaining of not descride method names\"))\n    end\n    if e.ht === Int\n        # update file to a stroducative, but is decould.\n        # xna i -GB =# [unsafe_color <c *has may num 20<11E 16/s\n        tuple | Expr(:(UnitLowerTriangular(transpose,(repl.ptr)))\n        dims = pipe_read(s,Int(a)...)\n    ex,0 + y.uilid_func & find_finwprevend(msg,:2)\n    ex = stage(c)\n    # uvvalue begin\n    end\nend"
},

{
    "location": "contributing.html#",
    "page": "Contributing & Help",
    "title": "Contributing & Help",
    "category": "page",
    "text": ""
},

{
    "location": "contributing.html#Contributing-1",
    "page": "Contributing & Help",
    "title": "Contributing",
    "category": "section",
    "text": "If you need help, please ask on the Julia forum or on Flux's Gitter.Right now, the best way to help out is to try out the examples and report any issues or missing features as you find them. The second best way is to help us spread the word, perhaps by starring the repo.If you're interested in hacking on Flux, most of the code is pretty straightforward. Adding new layer definitions or cost functions is simple using the Flux DSL itself, and things like data utilities and training processes are all plain Julia code. The compiler directory is a bit more involved and is documented in internals, but most changes won't need to touch that.If you get stuck or need anything, let us know!"
},

{
    "location": "internals.html#",
    "page": "Internals",
    "title": "Internals",
    "category": "page",
    "text": ""
},

{
    "location": "internals.html#Internals-1",
    "page": "Internals",
    "title": "Internals",
    "category": "section",
    "text": "[WIP]"
},

]}
