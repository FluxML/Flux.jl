using Documenter, Flux

makedocs(modules=Module[Flux],
         doctest=false, clean=true,
         format = :html,
         sitename="Flux Documentation",
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
