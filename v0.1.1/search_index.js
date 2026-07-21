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
    "text": "Flux is a high-level interface for machine learning, implemented in Julia.Flux aims to be an intuitive and powerful notation, close to the mathematics, that provides advanced features like auto-unrolling and closures. Simple models are trivial, while the most complex architectures are tractable, taking orders of magnitude less code than in other frameworks. Meanwhile, the Flux compiler provides excellent error messages and tools for debugging when things go wrong.So what's the catch? Flux is at an early \"working prototype\" stage; many things work but the API is still in a state of... well, it might change. If you're interested to find out what works, read on!"
},

{
    "location": "index.html#Where-do-I-start?-1",
    "page": "Home",
    "title": "Where do I start?",
    "category": "section",
    "text": "The examples are the best way to get a feel for how Flux looks. This a great way to start if you're a relative newbie to machine learning or neural networks; you should be able to get the examples running fairly easily.If you have more experience with ML, or you just don't want to see those digits again, check out the model building guide instead. The Guide attempts to motivate Flux's programming model and approach with examples. However, it also gets into advanced usage very quickly; it's not necessary to memorise all the details to use Flux effectively.The sections on Recurrence, Debugging and Batching best illustrate what makes Flux unique."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "... Charging Ion Capacitors ...Pkg.update()\nPkg.add(\"Flux.jl\")You'll also need a backend to run real training, if you don't have one already. Choose from MXNet or TensorFlow (MXNet is the recommended option if you're not sure):Pkg.add(\"MXNet\") # or \"TensorFlow\"\nPkg.test(\"Flux\") # Make sure everything installed properly"
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
    "location": "models/basics.html#The-Model-1",
    "page": "Model Building Basics",
    "title": "The Model",
    "category": "section",
    "text": "... Initialising Photon Beams ...The core concept in Flux is the model. A model (or \"layer\") is simply a function with parameters. For example, in plain Julia code, we could define the following function to represent a logistic regression (or simple neural network):W = randn(3,5)\nb = randn(3)\naffine(x) = W * x + b\n\nx1 = rand(5) # [0.581466,0.606507,0.981732,0.488618,0.415414]\ny1 = softmax(affine(x1)) # [0.32676,0.0974173,0.575823]affine is simply a function which takes some vector x1 and outputs a new one y1. For example, x1 could be data from an image and y1 could be predictions about the content of that image. However, affine isn't static. It has parameters W and b, and if we tweak those parameters we'll tweak the result – hopefully to make the predictions more accurate.This is all well and good, but we usually want to have more than one affine layer in our network; writing out the above definition to create new sets of parameters every time would quickly become tedious. For that reason, we want to use a template which creates these functions for us:affine1 = Affine(5, 5)\naffine2 = Affine(5, 5)\n\nsoftmax(affine1(x1)) # [0.167952, 0.186325, 0.176683, 0.238571, 0.23047]\nsoftmax(affine2(x1)) # [0.125361, 0.246448, 0.21966, 0.124596, 0.283935]We just created two separate Affine layers, and each contains its own (randomly initialised) version of W and b, leading to a different result when called with our data. It's easy to define templates like Affine ourselves (see templates), but Flux provides Affine out of the box, so we'll use that for now."
},

{
    "location": "models/basics.html#Combining-Models-1",
    "page": "Model Building Basics",
    "title": "Combining Models",
    "category": "section",
    "text": "... Inflating Graviton Zeppelins ...A more complex model usually involves many basic layers like affine, where we use the output of one layer as the input to the next:mymodel1(x) = softmax(affine2(σ(affine1(x))))\nmymodel1(x1) # [0.187935, 0.232237, 0.169824, 0.230589, 0.179414]This syntax is again a little unwieldy for larger networks, so Flux provides another template of sorts to create the function for us:mymodel2 = Chain(affine1, σ, affine2, softmax)\nmymodel2(x2) # [0.187935, 0.232237, 0.169824, 0.230589, 0.179414]mymodel2 is exactly equivalent to mymodel1 because it simply calls the provided functions in sequence. We don't have to predefine the affine layers and can also write this as:mymodel3 = Chain(\n  Affine(5, 5), σ,\n  Affine(5, 5), softmax)You now know enough to take a look at the logistic regression example, if you haven't already."
},

{
    "location": "models/basics.html#A-Function-in-Model's-Clothing-1",
    "page": "Model Building Basics",
    "title": "A Function in Model's Clothing",
    "category": "section",
    "text": "... Booting Dark Matter Transmogrifiers ...We noted above that a \"model\" is a function with some number of trainable parameters. This goes both ways; a normal Julia function like exp is effectively a model with 0 parameters. Flux doesn't care, and anywhere that you use one, you can use the other. For example, Chain will happily work with regular functions:foo = Chain(exp, sum, log)\nfoo([1,2,3]) == 3.408 == log(sum(exp([1,2,3])))"
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
    "text": "... Calculating Tax Expenses ...So how does the Affine template work? We don't want to duplicate the code above whenever we need more than one affine layer:W₁, b₁ = randn(...)\naffine₁(x) = W₁*x + b₁\nW₂, b₂ = randn(...)\naffine₂(x) = W₂*x + b₂\nmodel = Chain(affine₁, affine₂)Here's one way we could solve this: just keep the parameters in a Julia type, and define how that type acts as a function:type MyAffine\n  W\n  b\nend\n\n# Use the `MyAffine` layer as a model\n(l::MyAffine)(x) = l.W * x + l.b\n\n# Convenience constructor\nMyAffine(in::Integer, out::Integer) =\n  MyAffine(randn(out, in), randn(out))\n\nmodel = Chain(MyAffine(5, 5), MyAffine(5, 5))\n\nmodel(x1) # [-1.54458,0.492025,0.88687,1.93834,-4.70062]This is much better: we can now make as many affine layers as we want. This is a very common pattern, so to make it more convenient we can use the @net macro:@net type MyAffine\n  W\n  b\n  x -> x * W + b\nendThe function provided, x -> x * W + b, will be used when MyAffine is used as a model; it's just a shorter way of defining the (::MyAffine)(x) method above. (You may notice that W and x have swapped order in the model; this is due to the way batching works, which will be covered in more detail later on.)However, @net does not simply save us some keystrokes; it's the secret sauce that makes everything else in Flux go. For example, it analyses the code for the forward function so that it can differentiate it or convert it to a TensorFlow graph.The above code is almost exactly how Affine is defined in Flux itself! There's no difference between \"library-level\" and \"user-level\" models, so making your code reusable doesn't involve a lot of extra complexity. Moreover, much more complex models than Affine are equally simple to define."
},

{
    "location": "models/templates.html#Models-in-templates-1",
    "page": "Model Templates",
    "title": "Models in templates",
    "category": "section",
    "text": "@net models can contain sub-models as well as just array parameters:@net type TLP\n  first\n  second\n  function (x)\n    l1 = σ(first(x))\n    l2 = softmax(second(l1))\n  end\nendJust as above, this is roughly equivalent to writing:type TLP\n  first\n  second\nend\n\nfunction (self::TLP)(x)\n  l1 = σ(self.first(x))\n  l2 = softmax(self.second(l1))\nendClearly, the first and second parameters are not arrays here, but should be models themselves, and produce a result when called with an input array x. The Affine layer fits the bill, so we can instantiate TLP with two of them:model = TLP(Affine(10, 20),\n            Affine(20, 15))\nx1 = rand(20)\nmodel(x1) # [0.057852,0.0409741,0.0609625,0.0575354 ...You may recognise this as being equivalent toChain(\n  Affine(10, 20), σ\n  Affine(20, 15), softmax)given that it's just a sequence of calls. For simple networks Chain is completely fine, although the @net version is more powerful as we can (for example) reuse the output l1 more than once."
},

{
    "location": "models/templates.html#Constructors-1",
    "page": "Model Templates",
    "title": "Constructors",
    "category": "section",
    "text": "Affine has two array parameters, W and b. Just like any other Julia type, it's easy to instantiate an Affine layer with parameters of our choosing:a = Affine(rand(10, 20), rand(20))However, for convenience and to avoid errors, we'd probably rather specify the input and output dimension instead:a = Affine(10, 20)This is easy to implement using the usual Julia syntax for constructors:Affine(in::Integer, out::Integer) =\n  Affine(randn(in, out), randn(1, out))In practice, these constructors tend to take the parameter initialisation function as an argument so that it's more easily customisable, and use Flux.initn by default (which is equivalent to randn(...)/100). So Affine's constructor really looks like this:Affine(in::Integer, out::Integer; init = initn) =\n  Affine(init(in, out), init(1, out))"
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
    "text": "Recurrence is a first-class feature in Flux and recurrent models are very easy to build and use. Recurrences are often illustrated as cycles or self-dependencies in the graph; they can also be thought of as a hidden output from / input to the network. For example, for a sequence of inputs x1, x2, x3 ... we produce predictions as follows:y1 = f(W, x1) # `f` is the model, `W` represents the parameters\ny2 = f(W, x2)\ny3 = f(W, x3)\n...Each evaluation is independent and the prediction made for a given input will always be the same. That makes a lot of sense for, say, MNIST images, but less sense when predicting a sequence. For that case we introduce the hidden state:y1, s = f(W, x1, s)\ny2, s = f(W, x2, s)\ny3, s = f(W, x3, s)\n...The state s allows the prediction to depend not only on the current input x but also on the history of past inputs.The simplest recurrent network looks as follows in Flux, and it should be familiar if you've seen the equations defining an RNN before:@net type Recurrent\n  Wxy; Wyy; by\n  y\n  function (x)\n    y = tanh( x * Wxy + y{-1} * Wyy + by )\n  end\nendThe only difference from a regular feed-forward layer is that we create a variable y which is defined as depending on itself. The y{-1} syntax means \"take the value of y from the previous run of the network\".Using recurrent layers is straightforward and no different feedforard ones in terms of the Chain macro etc. For example:model = Chain(\n    Affine(784, 20), σ\n    Recurrent(20, 30),\n    Recurrent(30, 15))Before using the model we need to unroll it. This happens with the unroll function:unroll(model, 20)This call creates an unrolled, feed-forward version of the model which accepts N (= 20) inputs and generates N predictions at a time. Essentially, the model is replicated N times and Flux ties the hidden outputs y to hidden inputs.Here's a more complex recurrent layer, an LSTM, and again it should be familiar if you've seen the equations:@net type LSTM\n  Wxf; Wyf; bf\n  Wxi; Wyi; bi\n  Wxo; Wyo; bo\n  Wxc; Wyc; bc\n  y; state\n  function (x)\n    # Gates\n    forget = σ( x * Wxf + y{-1} * Wyf + bf )\n    input  = σ( x * Wxi + y{-1} * Wyi + bi )\n    output = σ( x * Wxo + y{-1} * Wyo + bo )\n    # State update and output\n    state′ = tanh( x * Wxc + y{-1} * Wyc + bc )\n    state  = forget .* state{-1} + input .* state′\n    y = output .* tanh(state)\n  end\nendThe only unfamiliar part is that we have to define all of the parameters of the LSTM upfront, which adds a few lines at the beginning.Flux's very mathematical notation generalises well to handling more complex models. For example, this neural translation model with alignment can be fairly straightforwardly, and recognisably, translated from the paper into Flux code:# A recurrent model which takes a token and returns a context-dependent\n# annotation.\n\n@net type Encoder\n  forward\n  backward\n  token -> hcat(forward(token), backward(token))\nend\n\nEncoder(in::Integer, out::Integer) =\n  Encoder(LSTM(in, out÷2), flip(LSTM(in, out÷2)))\n\n# A recurrent model which takes a sequence of annotations, attends, and returns\n# a predicted output token.\n\n@net type Decoder\n  attend\n  recur\n  state; y; N\n  function (anns)\n    energies = map(ann -> exp(attend(hcat(state{-1}, ann))[1]), seq(anns, N))\n    weights = energies./sum(energies)\n    ctx = sum(map((α, ann) -> α .* ann, weights, anns))\n    (_, state), y = recur((state{-1},y{-1}), ctx)\n    y\n  end\nend\n\nDecoder(in::Integer, out::Integer; N = 1) =\n  Decoder(Affine(in+out, 1),\n          unroll1(LSTM(in, out)),\n          param(zeros(1, out)), param(zeros(1, out)), N)\n\n# The model\n\nNalpha  =  5 # The size of the input token vector\nNphrase =  7 # The length of (padded) phrases\nNhidden = 12 # The size of the hidden state\n\nencode = Encoder(Nalpha, Nhidden)\ndecode = Chain(Decoder(Nhidden, Nhidden, N = Nphrase), Affine(Nhidden, Nalpha), softmax)\n\nmodel = Chain(\n  unroll(encode, Nphrase, stateful = false),\n  unroll(decode, Nphrase, stateful = false, seq = false))Note that this model excercises some of the more advanced parts of the compiler and isn't stable for general use yet."
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
    "text": "Existing machine learning frameworks and libraries represent batching, and other properties of data, only implicitly. Your machine learning data is a large N-dimensional array, which may have a shape like:100 × 50 × 256 × 256Typically, this might represent that you have (say) a batch of 100 samples, where each sample is a 50-long sequence of 256×256 images. This is great for performance, but array operations often become much more cumbersome as a result. Especially if you manipulate dimensions at runtime as an optimisation, debugging models can become extremely fiddly, with a proliferation of X × Y × Z arrays and no information about where they came from.Flux introduces a new approach where the batch dimension is represented explicitly as part of the data. For example:julia> xs = Batch([[1,2,3], [4,5,6]])\n2-element Batch of Vector{Int64}:\n [1,2,3]\n [4,5,6]Batches are represented the way we think about them; as an list of data points. We can do all the usual array operations with them, including getting the first with xs[1], iterating over them and so on. The trick is that under the hood, the data is batched into a single array:julia> rawbatch(xs)\n2×3 Array{Int64,2}:\n 1  2  3\n 4  5  6When we put a Batch object into a model, the model is ultimately working with a single array, which means there's no performance overhead and we get the full benefit of standard batching.Turning a set of vectors into a matrix is fairly easy anyway, so what's the big deal? Well, it gets more interesting as we start working with more complex data. Say we were working with 4×4 images:julia> xs = Batch([[1 2; 3 4], [5 6; 7 8]])\n2-element Flux.Batch of Array{Int64,2}:\n [1 2; 3 4]\n [5 6; 7 8]The raw batch array is much messier, and harder to recognise:julia> rawbatch(xs)\n2×2×2 Array{Int64,3}:\n[:, :, 1] =\n 1  3\n 5  7\n\n[:, :, 2] =\n 2  4\n 6  8Furthermore, because the batches acts like a list of arrays, we can use simple and familiar operations on it:julia> map(flatten, xs)\n2-element Array{Array{Int64,1},1}:\n [1,3,2,4]\n [5,7,6,8]flatten is simple enough over a single data point, but flattening a batched data set is more complex and you end up needing arcane array operations like mapslices. A Batch can just handle this for you for free, and more importantly it ensures that your operations are correct – that you haven't mixed up your batch and data dimensions, or used the wrong array op, and so on."
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
    "text": "model = Chain(Affine(10, 20), σ, Affine(20, 15), softmax)\nxs = rand(10)Currently, Flux's pure-Julia backend has no optimisations. This means that callingmodel(rand(10)) #> [0.0650, 0.0655, ...]directly won't have great performance. In order to run a computationally intensive training process, we rely on a backend like MXNet or TensorFlow.This is easy to do. Just call either mxnet or tf on a model to convert it to a model of that kind:mxmodel = mxnet(model)\nmxmodel(xs) #> [0.0650, 0.0655, ...]\n# or\ntfmodel = tf(model)\ntfmodel(xs) #> [0.0650, 0.0655, ...]These new models look and feel exactly like every other model in Flux, including returning the same result when you call them, and can be trained as usual using Flux.train!(). The difference is that the computation is being carried out by a backend, which will usually give a large speedup."
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
    "page": "Logistic Regression",
    "title": "Logistic Regression",
    "category": "page",
    "text": ""
},

{
    "location": "examples/logreg.html#Logistic-Regression-with-MNIST-1",
    "page": "Logistic Regression",
    "title": "Logistic Regression with MNIST",
    "category": "section",
    "text": "This walkthrough example will take you through writing a multi-layer perceptron that classifies MNIST digits with high accuracy.First, we load the data using the MNIST package:using Flux, MNIST\n\ndata = [(trainfeatures(i), onehot(trainlabel(i), 0:9)) for i = 1:60_000]\ntrain = data[1:50_000]\ntest = data[50_001:60_000]The only Flux-specific function here is onehot, which takes a class label and turns it into a one-hot-encoded vector that we can use for training. For example:julia> onehot(:b, [:a, :b, :c])\n3-element Array{Int64,1}:\n 0\n 1\n 0Otherwise, the format of the data is simple enough, it's just a list of tuples from input to output. For example:julia> data[1]\n([0.0,0.0,0.0, … 0.0,0.0,0.0],[0,0,0,0,0,1,0,0,0,0])data[1][1] is a 28*28 == 784 length vector (mostly zeros due to the black background) and data[1][2] is its classification.Now we define our model, which will simply be a function from one to the other.m = Chain(\n  Input(784),\n  Affine(128), relu,\n  Affine( 64), relu,\n  Affine( 10), softmax)\n\nmodel = tf(m)We can try this out on our data already:julia> model(data[1][1])\n10-element Array{Float64,1}:\n 0.10614  \n 0.0850447\n 0.101474\n ...The model gives a probability of about 0.1 to each class – which is a way of saying, \"I have no idea\". This isn't too surprising as we haven't shown it any data yet. This is easy to fix:Flux.train!(model, train, test, η = 1e-4)The training step takes about 5 minutes (to make it faster we can do smarter things like batching). If you run this code in Juno, you'll see a progress meter, which you can hover over to see the remaining computation time.Towards the end of the training process, Flux will have reported that the accuracy of the model is now about 90%. We can try it on our data again:10-element Array{Float32,1}:\n ...\n 5.11423f-7\n 0.9354     \n 3.1033f-5  \n 0.000127077\n ...Notice the class at 93%, suggesting our model is very confident about this image. We can use onecold to compare the true and predicted classes:julia> onecold(data[1][2], 0:9)\n5\n\njulia> onecold(model(data[1][1]), 0:9)\n5Success!"
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
    "text": "This walkthrough will take you through a model like that used in Karpathy's 2015 blog post, which can learn to generate text in the style of Shakespeare (or whatever else you may use as input). shakespeare_input.txt is here.using Flux\nimport StatsBase: wsampleFirstly, we define up front how many steps we want to unroll the RNN, and the number of data points to batch together. Then we create some functions to prepare our data, using Flux's built-in utilities.nunroll = 50\nnbatch = 50\n\ngetseqs(chars, alphabet) = sequences((onehot(Float32, char, alphabet) for char in chars), nunroll)\ngetbatches(chars, alphabet) = batches((getseqs(part, alphabet) for part in chunk(chars, nbatch))...)Because we want the RNN to predict the next letter at each iteration, our target data is simply our input data offset by one. For example, if the input is \"The quick brown fox\", the target will be \"he quick brown fox \". Each letter is one-hot encoded and sequences are batched together to create the training data.input = readstring(\"shakespeare_input.txt\")\nalphabet = unique(input)\nN = length(alphabet)\n\nXs, Ys = getbatches(input, alphabet), getbatches(input[2:end], alphabet)Creating the model and training it is straightforward:model = Chain(\n  Input(N),\n  LSTM(N, 256),\n  LSTM(256, 256),\n  Affine(256, N),\n  softmax)\n\nm = tf(unroll(model, nunroll))\n\n@time Flux.train!(m, Xs, Ys, η = 0.1, epoch = 1)Finally, we can sample the model. For sampling we remove the softmax from the end of the chain so that we can \"sharpen\" the resulting probabilities.function sample(model, n, temp = 1)\n  s = [rand(alphabet)]\n  m = tf(unroll(model, 1))\n  for i = 1:n\n    push!(s, wsample(alphabet, softmax(m(Seq((onehot(Float32, s[end], alphabet),)))[1]./temp)))\n  end\n  return string(s...)\nend\n\nsample(model[1:end-1], 100)sample then produces a string of Shakespeare-like text. This won't produce great results after only a single epoch (though they will be recognisably different from the untrained model). Going for 30 epochs or so produces good results.Trained on a dataset from base Julia, the network can produce code like:function show(io::IO, md::Githompty)\n    Buffer(jowerTriangular(inals[i], initabs_indices), characters, side, nextfloat(typeof(x)))\n    isnull(r) && return\n    start::I!\n    for j = 1:length(b,1)\n        a = s->cosvect(code)\n        return\n    end\n    indsERenv | maximum(func,lsg))\n    for i = 1:last(Abjelar) && fname (=== nothing)\n        throw(ArgumentError(\"read is declave non-fast-a/remaining of not descride method names\"))\n    end\n    if e.ht === Int\n        # update file to a stroducative, but is decould.\n        # xna i -GB =# [unsafe_color <c *has may num 20<11E 16/s\n        tuple | Expr(:(UnitLowerTriangular(transpose,(repl.ptr)))\n        dims = pipe_read(s,Int(a)...)\n    ex,0 + y.uilid_func & find_finwprevend(msg,:2)\n    ex = stage(c)\n    # uvvalue begin\n    end\nend"
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
