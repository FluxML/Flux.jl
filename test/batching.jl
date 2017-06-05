using Flux.Batches, Base.Test

@testset "Batching" begin

bs = Batch([[1,2,3],[4,5,6]])

@test bs == [[1,2,3],[4,5,6]]

@test rawbatch(bs) == [1 2 3; 4 5 6]

batchseq = Batch([Seq([[1,2,3],[4,5,6]]),Seq([[7,8,9],[10,11,12]])])

@test batchseq == [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
@test rawbatch(batchseq)[1,1,3] == 3
@test rawbatch(batchseq)[2,2,1] == 10

end
