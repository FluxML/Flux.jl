using CUDA, Enzyme
x = CUDA.ones(2) 
dx = CUDA.zeros(2)
autodiff(Reverse, sum, Active, Duplicated(x, dx))
