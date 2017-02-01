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
    "location": "models/basics.html#The-Template-1",
    "page": "Model Building Basics",
    "title": "The Template",
    "category": "section",
    "text": "... Calculating Tax Expenses ...So how does the Affine template work? We don't want to duplicate the code above whenever we need more than one affine layer:W₁, b₁ = randn(...)\naffine₁(x) = W₁*x + b₁\nW₂, b₂ = randn(...)\naffine₂(x) = W₂*x + b₂\nmodel = Chain(affine₁, affine₂)Here's one way we could solve this: just keep the parameters in a Julia type, and define how that type acts as a function:type MyAffine\n  W\n  b\nend\n\n# Use the `MyAffine` layer as a model\n(l::MyAffine)(x) = l.W * x + l.b\n\n# Convenience constructor\nMyAffine(in::Integer, out::Integer) =\n  MyAffine(randn(out, in), randn(out))\n\nmodel = Chain(MyAffine(5, 5), MyAffine(5, 5))\n\nmodel(x1) # [-1.54458,0.492025,0.88687,1.93834,-4.70062]This is much better: we can now make as many affine layers as we want. This is a very common pattern, so to make it more convenient we can use the @net macro:@net type MyAffine\n  W\n  b\n  x -> W * x + b\nendThe function provided, x -> W * x + b, will be used when MyAffine is used as a model; it's just a shorter way of defining the (::MyAffine)(x) method above.However, @net does not simply save us some keystrokes; it's the secret sauce that makes everything else in Flux go. For example, it analyses the code for the forward function so that it can differentiate it or convert it to a TensorFlow graph.The above code is almost exactly how Affine is defined in Flux itself! There's no difference between \"library-level\" and \"user-level\" models, so making your code reusable doesn't involve a lot of extra complexity. Moreover, much more complex models than Affine are equally simple to define."
},

{
    "location": "models/basics.html#Sub-Templates-1",
    "page": "Model Building Basics",
    "title": "Sub-Templates",
    "category": "section",
    "text": "@net models can contain sub-models as well as just array parameters:@net type TLP\n  first\n  second\n  function (x)\n    l1 = σ(first(x))\n    l2 = softmax(second(l1))\n  end\nendJust as above, this is roughly equivalent to writing:type TLP\n  first\n  second\nend\n\nfunction (self::TLP)(x)\n  l1 = σ(self.first)\n  l2 = softmax(self.second(l1))\nendClearly, the first and second parameters are not arrays here, but should be models themselves, and produce a result when called with an input array x. The Affine layer fits the bill so we can instantiate TLP with two of them:model = TLP(Affine(10, 20),\n            Affine(20, 15))\nx1 = rand(20)\nmodel(x1) # [0.057852,0.0409741,0.0609625,0.0575354 ...You may recognise this as being equivalent toChain(\n  Affine(10, 20), σ\n  Affine(20, 15)), softmaxgiven that it's just a sequence of calls. For simple networks Chain is completely fine, although the @net version is more powerful as we can (for example) reuse the output l1 more than once."
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
    "text": "[WIP]"
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
    "text": "[WIP]"
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
