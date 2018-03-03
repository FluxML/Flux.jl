using Flux.Data
using Base.Test

@test cmudict()["CATASTROPHE"] == :[K,AH0,T,AE1,S,T,R,AH0,F,IY0].args

@test length(CMUDict.phones()) == 39

@test length(CMUDict.symbols()) == 84
