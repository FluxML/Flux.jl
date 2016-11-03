using Flux

# Traditional Approach

# 100 samples of sequences of 15 28×28 3-colour images
rand(100, 15, 28, 28, 3)

# Basic Batching

data = Batch([collect(reshape(9(i-1):9i-1, 3, 3)) for i = 1:10])

Batch(flatten.(data))

data |> structure

Batch(flatten.(data)) |> structure

# Nested Batching

# DNA seqence, encoded as a list of [A, T, G, C]
x1 = Seq([[0,1,0,0], [1,0,0,0], [0,0,0,1]])
x2 = Seq([[0,0,1,0], [0,0,0,1], [0,0,1,0]])

data = Batch([x1, x2])

data |> structure
