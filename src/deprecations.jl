# v0.11 deprecations
@deprecate poisson poisson_loss false
@deprecate hinge hinge_loss false
@deprecate squared_hinge squared_hinge_loss false
@deprecate binarycrossentropy(ŷ, y) Flux.Losses.binarycrossentropy(ŷ, y, agg=identity) false
@deprecate logitbinarycrossentropy(ŷ, y) Flux.Losses.logitbinarycrossentropy(ŷ, y, agg=identity) false
@deprecate normalise(x) normalise(x, dims=1) false
