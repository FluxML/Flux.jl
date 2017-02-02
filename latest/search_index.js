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
    "text": "Flux is a high-level interface for machine learning, implemented in Julia.Flux aims to be an intuitive and powerful notation, close to the mathematics, that provides advanced features like auto-unrolling and closures. Simple models are trivial, while the most complex architectures are tractable, taking orders of magnitude less code than in other frameworks. Meanwhile, the Flux compiler provides excellent error messages and tools for debugging when things go wrong.So what's the catch? Flux is at an early \"working prototype\" stage; many things work but the API is still in a state of... well, it might change. Also, this documentation is pretty incomplete.If you're interested to find out what does work, read on!"
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "... Charging Ion Capacitors ...Pkg.clone(\"https://github.com/MikeInnes/DataFlow.jl\")\nPkg.clone(\"https://github.com/MikeInnes/Flux.jl\")\nusing FluxYou'll also need a backend to run real training, if you don't have one already. Choose from MXNet or TensorFlow (MXNet is the recommended option if you're not sure):Pkg.add(\"MXNet\") # or \"TensorFlow\""
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
    "text": "... Initialising Photon Beams ...The core concept in Flux is the model. A model (or \"layer\") is simply a function with parameters. For example, in plain Julia code, we could define the following function to represent a logistic regression (or simple neural network):W = randn(3,5)\nb = randn(3)\naffine(x) = W * x + b\n\nx1 = rand(5) # [0.581466,0.606507,0.981732,0.488618,0.415414]\ny1 = softmax(affine(x1)) # [0.32676,0.0974173,0.575823]affine is simply a function which takes some vector x1 and outputs a new one y1. For example, x1 could be data from an image and y1 could be predictions about the content of that image. However, affine isn't static. It has parameters W and b, and if we tweak those parameters we'll tweak the result – hopefully to make the predictions more accurate.This is all well and good, but we usually want to have more than one affine layer in our network; writing out the above definition to create new sets of parameters every time would quickly become tedious. For that reason, we want to use a template which creates these functions for us:affine1 = Affine(5, 5)\naffine2 = Affine(5, 5)\n\nsoftmax(affine1(x1)) # [0.167952, 0.186325, 0.176683, 0.238571, 0.23047]\nsoftmax(affine2(x1)) # [0.125361, 0.246448, 0.21966, 0.124596, 0.283935]We just created two separate Affine layers, and each contains its own version of W and b, leading to a different result when called with our data. It's easy to define templates like Affine ourselves (see The Template), but Flux provides Affine out of the box, so we'll use that for now."
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
    "text": "... Calculating Tax Expenses ...So how does the Affine template work? We don't want to duplicate the code above whenever we need more than one affine layer:W₁, b₁ = randn(...)\naffine₁(x) = W₁*x + b₁\nW₂, b₂ = randn(...)\naffine₂(x) = W₂*x + b₂\nmodel = Chain(affine₁, affine₂)Here's one way we could solve this: just keep the parameters in a Julia type, and define how that type acts as a function:type MyAffine\n  W\n  b\nend\n\n# Use the `MyAffine` layer as a model\n(l::MyAffine)(x) = l.W * x + l.b\n\n# Convenience constructor\nMyAffine(in::Integer, out::Integer) =\n  MyAffine(randn(out, in), randn(out))\n\nmodel = Chain(MyAffine(5, 5), MyAffine(5, 5))\n\nmodel(x1) # [-1.54458,0.492025,0.88687,1.93834,-4.70062]This is much better: we can now make as many affine layers as we want. This is a very common pattern, so to make it more convenient we can use the @net macro:@net type MyAffine\n  W\n  b\n  x -> W * x + b\nendThe function provided, x -> W * x + b, will be used when MyAffine is used as a model; it's just a shorter way of defining the (::MyAffine)(x) method above.However, @net does not simply save us some keystrokes; it's the secret sauce that makes everything else in Flux go. For example, it analyses the code for the forward function so that it can differentiate it or convert it to a TensorFlow graph.The above code is almost exactly how Affine is defined in Flux itself! There's no difference between \"library-level\" and \"user-level\" models, so making your code reusable doesn't involve a lot of extra complexity. Moreover, much more complex models than Affine are equally simple to define."
},

{
    "location": "models/templates.html#Models-in-templates-1",
    "page": "Model Templates",
    "title": "Models in templates",
    "category": "section",
    "text": "@net models can contain sub-models as well as just array parameters:@net type TLP\n  first\n  second\n  function (x)\n    l1 = σ(first(x))\n    l2 = softmax(second(l1))\n  end\nendJust as above, this is roughly equivalent to writing:type TLP\n  first\n  second\nend\n\nfunction (self::TLP)(x)\n  l1 = σ(self.first(x))\n  l2 = softmax(self.second(l1))\nendClearly, the first and second parameters are not arrays here, but should be models themselves, and produce a result when called with an input array x. The Affine layer fits the bill so we can instantiate TLP with two of them:model = TLP(Affine(10, 20),\n            Affine(20, 15))\nx1 = rand(20)\nmodel(x1) # [0.057852,0.0409741,0.0609625,0.0575354 ...You may recognise this as being equivalent toChain(\n  Affine(10, 20), σ\n  Affine(20, 15), softmax)given that it's just a sequence of calls. For simple networks Chain is completely fine, although the @net version is more powerful as we can (for example) reuse the output l1 more than once."
},

{
    "location": "models/templates.html#Constructors-1",
    "page": "Model Templates",
    "title": "Constructors",
    "category": "section",
    "text": "Affine has two array parameters, W and b. Just like any other Julia type, it's easy to instantiate an Affine layer with parameters of our choosing:a = Affine(rand(10, 20), rand(20))However, for convenience and to avoid errors, we'd probably rather specify the input and output dimension instead:a = Affine(10, 20)This is easy to implement using the usual Julia syntax for constructors:Affine(in::Integer, out::Integer) = Affine(randn(in, out), randn(1, out))In practice, these constructors tend to take the parameter initialisation function as an argument so that it's more easily customisable, and use Flux.initn by default (which is equivalent to randn()/100). So Affine's constructor really looks like this:Affine(in::Integer, out::Integer; init = initn) =\n  Affine(init(in, out), init(1, out))"
},

{
    "location": "models/templates.html#Supported-syntax-1",
    "page": "Model Templates",
    "title": "Supported syntax",
    "category": "section",
    "text": "The syntax used to define a forward pass like x -> W*x + b behaves exactly like Julia code for the most part. However, it's important to remember that it's defining a dataflow graph, not a general Julia expression. In practice this means that anything side-effectful, or things like control flow and printlns, won't work as expected. In future we'll continue expand support for Julia syntax and features."
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
    "text": "Let's take our two-layer perceptron as an example again, running on MXNet:@net type TLP\n  first\n  second\n  function (x)\n    l1 = σ(first(x))\n    l2 = softmax(second(l1))\n  end\nend\n\nmodel = TLP(Affine(10, 20), Affine(21, 15))\n\nmxmodel = mxnet(model, (1, 20))Unfortunately, this model has a (fairly obvious) typo, which means that the code above won't run. Instead we get an error message:InferShape Error in dot5: [20:37:39] src/operator/./matrix_op-inl.h:271: Check failed: (lshape[1]) == (rshape[0]) dot shape error: (15,21) X (20,1)\n in Flux.Affine at affine.jl:8\n in TLP at test.jl:6\n in mxnet(::TLP, ::Tuple{Int64,Int64}) at model.jl:40\n in mxnet(::TLP, ::Vararg{Any,N} where N) at backend.jl:20Most frameworks would only give the error message here – not so helpful if you have thousands of nodes in your computational graph. However, Flux is able to give good error reports even when no Julia code has been run, e.g. when running on a backend like MXNet. This enables us to pinpoint the source of the error very quickly even in a large model.In this case, we can immediately see that the error occurred within an Affine layer. There are two such layers, but this one was called from the second line of TLP, so it must be the second Affine layer we defined. The layer expected an input of length 21 but got 20 instead.Of course, often a stack trace isn't enough to figure out the source of an error. Another option is to simply step through the execution of the model using Gallium. While handy, however, stepping isn't always the best way to get a \"bird's eye view\" of the code. For that, Flux provides a macro called @shapes:julia> @shapes model(rand(5,10))\n\n# /Users/mike/test.jl, line 18:\ngull = σ(Affine(10, 20)(Input()[1]::(5,10))::(5,20))::(5,20)\n# /Users/mike/.julia/v0.6/Flux/src/layers/affine.jl, line 8:\nlobster = gull * _::(21,15) + _::(1,15)\n# /Users/mike/test.jl, line 19:\nraven = softmax(lobster)This is a lot like Julia's own code_warntype; but instead of annotating expressions with types, we display their shapes. As a lowered form it has some quirks; input arguments are represented by Input()[N] and parameters by an underscore.This makes the problem fairly obvious. We tried to multiply the output of the first layer (5, 20) by a parameter (21, 15); the inner dimensions should have been equal.Notice that while the first Affine layer is displayed as-is, the second was inlined and we see a reference to where the W * x + b line was defined in Flux's source code. In this way Flux makes it easy to drill down into problem areas, without showing you the full graph of thousands of nodes at once.With the typo fixed, the output of @shapes looks as follows:# /Users/mike/test.jl, line 18:\nopossum = σ(Affine(10, 20)(Input()[1]::(5,10))::(5,20))::(5,20)\n# /Users/mike/test.jl, line 19:\nwren = softmax(Affine(20, 15)(opossum)::(5,15))::(5,15)"
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
    "text": "[WIP]"
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
    "text": "[WIP]"
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
