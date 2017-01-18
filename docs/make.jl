using Documenter, Flux

makedocs(modules=[Flux],
         doctest = false,
         format = :html,
         analytics = "UA-36890222-9",
         sitename = "Flux",
         assets = ["../flux.css"],
         pages = ["Home" => "index.md",
                  "First Steps" => "manual/basics.md",
                  "Recurrence" => "manual/recurrent.md",
                  "Debugging" => "manual/debugging.md",
                  "In Action" => [
                    "Logistic Regression" => "examples/logreg.md"]
                  "Contributing & Help" => "contributing.md",
                  "Internals"])

deploydocs(
   repo = "github.com/MikeInnes/Flux.jl.git",
   target = "build",
   osname = "linux",
   julia = "0.5",
   deps = nothing,
   make = nothing)
