using Documenter, Flux

makedocs(modules=Module[Flux],
         doctest=false, clean=false,
         format = :html,
         analytics = "UA-36890222-9",
         sitename="Flux",
         pages = [
           "Home" => "index.md",
         ])

deploydocs(
   repo = "github.com/MikeInnes/Flux.jl.git",
   target = "build",
   osname = "linux",
   julia = "0.5",
   deps = nothing,
   make = nothing)
