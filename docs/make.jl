using Documenter, Flux

makedocs(modules=[Flux],
         doctest = false,
         format = :html,
         analytics = "UA-36890222-9",
         sitename = "Flux",
         assets = ["../flux.css"],
         pages = ["Home" => "index.md",
                  "Getting Started" => [
                    "Basics" => "manual/basics.md",
                    "Custom Layers" => "manual/custom.md",
                    "Recurrence" => "manual/recurrent.md",
                    "Debugging" => "manual/debugging.md"],
                  "Contributing & Help" => "contributing.md"])

deploydocs(
   repo = "github.com/MikeInnes/Flux.jl.git",
   target = "build",
   osname = "linux",
   julia = "0.5",
   deps = nothing,
   make = nothing)
