bs = Batch([[1,2,3],[4,5,6]])

@test typeof(bs) <: Batch

@test bs == [[1,2,3],[4,5,6]]

@test rawbatch(bs) == [1 2 3; 4 5 6]
