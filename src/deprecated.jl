import Base: @deprecate

#### remove in v 0.11   #####
@deprecate param(x) x
@deprecate data(x) x

@deprecate mapleaves(f, x) fmap(f, x)

macro treelike(args...)
    functorm(args...)
end
#############################

  