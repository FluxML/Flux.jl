module Sentiment

using ZipFile
using ..Data: deps

function load()
  isfile(deps("sentiment.zip")) ||
    download("https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip",
             deps("sentiment.zip"))
  return
end

getfile(r, name) = r.files[findfirst(x -> x.name == name, r.files)]

function getfile(name)
  r = ZipFile.Reader(deps("sentiment.zip"))
  text = readstring(getfile(r, "trees/$name"))
  close(r)
  return text
end

using ..Flux.Batches

totree_(n, w) = Tree{Any}((parse(Int, n), w))
totree_(n, a, b) = Tree{Any}((parse(Int, n), nothing), totree(a), totree(b))
totree(t::Expr) = totree_(t.args...)

function parsetree(s)
  s = replace(s, r"\$", s -> "\\\$")
  s = replace(s, r"[^\s\(\)]+", s -> "\"$s\"")
  s = replace(s, " ", ", ")
  return totree(parse(s))
end

function gettrees(name)
  load()
  ss = split(getfile("$name.txt"), '\n', keep = false)
  return parsetree.(ss)
end

train() = gettrees("train")
test() = gettrees("test")
dev() = gettrees("dev")

end
