# =======================================================
#                     NCCL backend
#                    RUN THIS WITH 
# `mpiexecjl --project=@. -n 3 julia distributed_NCCL.jl`
# =======================================================

using Flux, MPI, NCCL

DistributedUtils.initialize(NCCLBackend)

backend = DistributedUtils.get_distributed_backend(NCCLBackend) 
rank = DistributedUtils.local_rank(backend)

device = Flux.get_device()
model = Chain(Dense(1 => 256, tanh), Dense(256 => 1)) |> gpu
x = rand(Float32, 1, 16) |> gpu
y = x .^ 3

model = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model); root=0) 

opt = DistributedUtils.DistributedOptimizer(backend, Optimisers.Adam(0.001f0))
st_opt = Optimisers.setup(opt, model)
st_opt = DistributedUtils.synchronize!!(backend, st_opt; root=0) 

loss(model) = mean((model(x) .- y).^2)
g_ = gradient(m -> loss(m), model)[1] 
Optimisers.update!(st_opt, model, g_)

for epoch in 1:100
  global model, st_opt
  l, back = Zygote.pullback(loss, model)
  println("Epoch $epoch: Loss $l")
  g = back(one(l))[1]
  st_opt, model = Optimisers.update(st_opt, model, g)
end
