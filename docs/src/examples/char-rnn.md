# Char RNN

This walkthrough will take you through a model like that used in [Karpathy's 2015 blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), which can learn to generate text in the style of Shakespeare (or whatever else you may use as input). `shakespeare_input.txt` is [here](http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt).

```julia
using Flux
import StatsBase: wsample
```

Firstly, we define up front how many steps we want to unroll the RNN, and the number of data points to batch together. Then we create some functions to prepare our data, using Flux's built-in utilities.

```julia
nunroll = 50
nbatch = 50

getseqs(chars, alphabet) = sequences((onehot(Float32, char, alphabet) for char in chars), nunroll)
getbatches(chars, alphabet) = batches((getseqs(part, alphabet) for part in chunk(chars, nbatch))...)
```

Because we want the RNN to predict the next letter at each iteration, our target data is simply our input data offset by one. For example, if the input is "The quick brown fox", the target will be "he quick brown fox ". Each letter is one-hot encoded and sequences are batched together to create the training data.

```julia
input = readstring("shakespeare_input.txt")
alphabet = unique(input)
N = length(alphabet)

Xs, Ys = getbatches(input, alphabet), getbatches(input[2:end], alphabet)
```

Creating the model and training it is straightforward:

```julia
model = Chain(
  Input(N),
  LSTM(N, 256),
  LSTM(256, 256),
  Affine(256, N),
  softmax)

m = tf(unroll(model, nunroll))

@time Flux.train!(m, Xs, Ys, η = 0.1, epoch = 1)
```

Finally, we can sample the model. For sampling we remove the `softmax` from the end of the chain so that we can "sharpen" the resulting probabilities.

```julia
function sample(model, n, temp = 1)
  s = [rand(alphabet)]
  m = tf(unroll(model, 1))
  for i = 1:n
    push!(s, wsample(alphabet, softmax(m(Seq((onehot(Float32, s[end], alphabet),)))[1]./temp)))
  end
  return string(s...)
end

sample(model[1:end-1], 100)
```

`sample` then produces a string of Shakespeare-like text. This won't produce great results after only a single epoch (though they will be recognisably different from the untrained model). Going for 30 epochs or so produces good results.
