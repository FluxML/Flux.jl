# v0.11 deprecations
@deprecate poisson poisson_loss
@deprecate hinge hinge_loss
@deprecate squared_hinge squared_hinge_loss
@deprecate binarycrossentropy(ŷ, y) bce_loss(ŷ, y, agg=identity)
@deprecate logitbinarycrossentropy(ŷ, y) logitbce_loss(ŷ, y, agg=identity)
@deprecate normalise(x) normalise(x, dims=1) 
