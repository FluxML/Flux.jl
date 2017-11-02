module Sentiment

using ..Data: deps

function load()
  isfile(deps("stanfordSentimentTreebank.zip")) ||
    download("http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip",
             deps("stanfordSentimentTreebank.zip"))
  return
end

getfile(r, name) = r.files[findfirst(x -> x.name == name, r.files)]

function loadtext()
  r = ZipFile.Reader(deps("stanfordSentimentTreebank.zip"))
  sentences = readstring(getfile(r, "stanfordSentimentTreebank/datasetSentences.txt"))
  close(r)
  return sentences
end

end
