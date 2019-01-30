module Sentiment

using ZipFile
using ..Data: deps, download_and_verify

function load()
  isfile(deps("sentiment.zip")) && return
  @info "Downloading sentiment treebank dataset"
  download_and_verify("https://cache.julialang.org/https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip",
           deps("sentiment.zip"), "5c613a4f673fc74097d523a2c83f38e0cc462984d847b82c7aaf36b01cbbbfcc")
end

getfile(r, name) = r.files[findfirst(x -> x.name == name, r.files)]

function getfile(name)
  r = ZipFile.Reader(deps("sentiment.zip"))
  text = read(getfile(r, "trees/$name"), String)
  close(r)
  return text
end

using ..Data: Tree

totree_(n, w) = Tree{Any}((parse(Int, n), w))
totree_(n, a, b) = Tree{Any}((parse(Int, n), nothing), totree(a), totree(b))
totree(t::Expr) = totree_(t.args...)

function parsetree(s)
  s = replace(s, "\\" => "")
  s = replace(s, "\$" => "\\\$")
  s = replace(s, r"[^ \n\(\)]+" => s -> "\"$s\"")
  s = replace(s, " " => ", ")
  return totree(Meta.parse(s))
end

function gettrees(name)
  load()
  ss = split(getfile("$name.txt"), '\n', keepempty = false)
  return parsetree.(ss)
end

train() = gettrees("train")
test() = gettrees("test")
dev() = gettrees("dev")

end
