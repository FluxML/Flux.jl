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
    "text": "Flux is a high-level interface for machine learning, implemented in Julia.Flux aims to be an intuitive and powerful notation, close to the mathematics, that provides advanced features like auto-unrolling and closures. Simple models are trivial, while the most complex architectures are tractable, taking orders of magnitude less code than in other frameworks. Meanwhile, the Flux compiler provides excellent error messages and tools for debugging when things go wrong.So what's the catch? Flux is at an early \"working prototype\" stage; many things work but the API is still in a state of... well, it might change.If you're interested to find out what does work, read on!"
},

{
    "location": "manual/basics.html#",
    "page": "Basics",
    "title": "Basics",
    "category": "page",
    "text": ""
},

{
    "location": "manual/basics.html#Basic-Usage-1",
    "page": "Basics",
    "title": "Basic Usage",
    "category": "section",
    "text": ""
},

{
    "location": "manual/basics.html#Installation-1",
    "page": "Basics",
    "title": "Installation",
    "category": "section",
    "text": "Pkg.clone(\"https://github.com/MikeInnes/DataFlow.jl\")\nPkg.clone(\"https://github.com/MikeInnes/Flux.jl\")"
},

{
    "location": "manual/basics.html#The-Model-1",
    "page": "Basics",
    "title": "The Model",
    "category": "section",
    "text": "Charging Ion Capacitors...The core concept in Flux is that of the model. A model is simply a function with parameters. In Julia, we might define the following function:W = randn(3,5)\nb = randn(3)\naffine(x) = W*x + b\n\nx1 = randn(5)\naffine(x1)\n> 3-element Array{Float64,1}:\n   -0.0215644\n   -4.07343  \n    0.312591"
},

{
    "location": "manual/basics.html#An-MNIST-Example-1",
    "page": "Basics",
    "title": "An MNIST Example",
    "category": "section",
    "text": ""
},

{
    "location": "manual/custom.html#",
    "page": "Custom Layers",
    "title": "Custom Layers",
    "category": "page",
    "text": ""
},

{
    "location": "manual/custom.html#Custom-Layers-1",
    "page": "Custom Layers",
    "title": "Custom Layers",
    "category": "section",
    "text": "[WIP]"
},

{
    "location": "manual/recurrent.html#",
    "page": "Recurrence",
    "title": "Recurrence",
    "category": "page",
    "text": ""
},

{
    "location": "manual/recurrent.html#Recurrent-Models-1",
    "page": "Recurrence",
    "title": "Recurrent Models",
    "category": "section",
    "text": "[WIP]"
},

{
    "location": "manual/debugging.html#",
    "page": "Debugging",
    "title": "Debugging",
    "category": "page",
    "text": ""
},

{
    "location": "manual/debugging.html#Debugging-Models-1",
    "page": "Debugging",
    "title": "Debugging Models",
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

]}
