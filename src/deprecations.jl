# v0.11 deprecations
@deprecate poisson poisson_loss false
@deprecate hinge hinge_loss false
@deprecate squared_hinge squared_hinge_loss false
@deprecate normalise(x) normalise(x, dims=1) false

@deprecate binarycrossentropy(ŷ, y) Losses.binarycrossentropy(ŷ, y, agg=identity) false
@deprecate logitbinarycrossentropy(ŷ, y) Losses.logitbinarycrossentropy(ŷ, y, agg=identity) false

function Broadcast.broadcasted(::typeof(binarycrossentropy), ŷ, y)
    @warn "binarycrossentropy.(ŷ, y) is deprecated, use Losses.binarycrossentropy(ŷ, y, agg=identity) instead"
    Losses.binarycrossentropy(ŷ, y, agg=identity)
end

function Broadcast.broadcasted(::typeof(logitbinarycrossentropy), ŷ, y)
    @warn "logitbinarycrossentropy.(ŷ, y) is deprecated, use Losses.logitbinarycrossentropy(ŷ, y, agg=identity) instead"
    Losses.logitbinarycrossentropy(ŷ, y, agg=identity)
end