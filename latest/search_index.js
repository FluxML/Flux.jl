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
    "text": "Install Julia 0.6.0 or later, if you haven't already.Pkg.add(\"Flux\")\nPkg.test(\"Flux\") # Check things installed correctlyStart with the basics. The model zoo is also a good starting point for many common kinds of models."
},

{
    "location": "models/basics.html#",
    "page": "Basics",
    "title": "Basics",
    "category": "page",
    "text": ""
},

{
    "location": "models/basics.html#Taking-Gradients-1",
    "page": "Basics",
    "title": "Taking Gradients",
    "category": "section",
    "text": "Consider a simple linear regression, which tries to predict an output array y from an input x. (It's a good idea to follow this example in the Julia repl.)W = rand(2, 5)\nb = rand(2)\n\npredict(x) = W*x .+ b\nloss(x, y) = sum((predict(x) .- y).^2)\n\nx, y = rand(5), rand(2) # Dummy data\nloss(x, y) # ~ 3To improve the prediction we can take the gradients of W and b with respect to the loss function and perform gradient descent. We could calculate gradients by hand, but Flux will do it for us if we tell it that W and b are trainable parameters.using Flux.Tracker: param, back!, data, grad\n\nW = param(W)\nb = param(b)\n\nl = loss(x, y)\n\nback!(l)loss(x, y) returns the same number, but it's now a tracked value that records gradients as it goes along. Calling back! then calculates the gradient of W and b. We can see what this gradient is, and modify W to train the model.grad(W)\n\nW.data .-= grad(W)\n\nloss(x, y) # ~ 2.5The loss has decreased a little, meaning that our prediction x is closer to the target y. If we have some data we can already try training the model.All deep learning in Flux, however complex, is a simple generalisation of this example. Of course, not all models look like this – they might have millions of parameters or complex control flow, and Flux provides ways to manage this complexity. Let's see what that looks like."
},

{
    "location": "models/basics.html#Building-Layers-1",
    "page": "Basics",
    "title": "Building Layers",
    "category": "section",
    "text": "It's common to create more complex models than the linear regression above. For example, we might want to have two linear layers with a nonlinearity like sigmoid (σ) in between them. In the above style we could write this as:W1 = param(rand(3, 5))\nb1 = param(rand(3))\nlayer1(x) = W1 * x .+ b1\n\nW2 = param(rand(2, 3))\nb2 = param(rand(2))\nlayer2(x) = W2 * x .+ b2\n\nmodel(x) = layer2(σ.(layer1(x)))\n\nmodel(rand(5)) # => 2-element vectorThis works but is fairly unwieldy, with a lot of repetition – especially as we add more layers. One way to factor this out is to create a function that returns linear layers.function linear(in, out)\n  W = param(randn(out, in))\n  b = param(randn(out))\n  x -> W * x .+ b\nend\n\nlinear1 = linear(5, 3) # we can access linear1.W etc\nlinear2 = linear(3, 2)\n\nmodel(x) = linear2(σ.(linear1(x)))\n\nmodel(x) # => 2-element vectorAnother (equivalent) way is to create a struct that explicitly represents the affine layer.struct Affine\n  W\n  b\nend\n\nAffine(in::Integer, out::Integer) =\n  Affine(param(randn(out, in)), param(randn(out)))\n\n# Overload call, so the object can be used as a function\n(m::Affine)(x) = m.W * x .+ m.b\n\na = Affine(10, 5)\n\na(rand(10)) # => 5-element vectorCongratulations! You just built the Dense layer that comes with Flux. Flux has many interesting layers available, but they're all things you could have built yourself very easily.(There is one small difference with Dense – for convenience it also takes an activation function, like Dense(10, 5, σ).)"
},

{
    "location": "models/basics.html#Stacking-It-Up-1",
    "page": "Basics",
    "title": "Stacking It Up",
    "category": "section",
    "text": "It's pretty common to write models that look something like:layer1 = Dense(10, 5, σ)\n# ...\nmodel(x) = layer3(layer2(layer1(x)))For long chains, it might be a bit more intuitive to have a list of layers, like this:using Flux\n\nlayers = [Dense(10, 5, σ), Dense(5, 2), softmax]\n\nmodel(x) = foldl((x, m) -> m(x), x, layers)\n\nmodel(rand(10)) # => 2-element vectorHandily, this is also provided for in Flux:model2 = Chain(\n  Dense(10, 5, σ),\n  Dense(5, 2),\n  softmax)\n\nmodel2(rand(10)) # => 2-element vectorThis quickly starts to look like a high-level deep learning library; yet you can see how it falls out of simple abstractions, and we lose none of the power of Julia code.A nice property of this approach is that because \"models\" are just functions (possibly with trainable parameters), you can also see this as simple function composition.m = Dense(5, 2) ∘ Dense(10, 5, σ)\n\nm(rand(10))Likewise, Chain will happily work with any Julia function.m = Chain(x -> x^2, x -> x+1)\n\nm(5) # => 26"
},

{
    "location": "models/recurrence.html#",
    "page": "Recurrence",
    "title": "Recurrence",
    "category": "page",
    "text": ""
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
    "text": "If you need help, please ask on the Julia forum, the slack (channel #machine-learning), or Flux's Gitter.Right now, the best way to help out is to try out the examples and report any issues or missing features as you find them. The second best way is to help us spread the word, perhaps by starring the repo.If you're interested in hacking on Flux, most of the code is pretty straightforward. Adding new layer definitions or cost functions is simple using the Flux DSL itself, and things like data utilities and training processes are all plain Julia code.If you get stuck or need anything, let us know!"
},

]}
