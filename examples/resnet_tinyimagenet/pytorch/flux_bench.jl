# Timed driver for the Flux ResNet-18 / Tiny-ImageNet example, mirroring the PyTorch
# `--benchmark-epochs` run: N timed training epochs, no evaluation — just the per-epoch training
# wall-time and throughput. Reuses the model / data / `loss_fn` / `trainstep!` loop from the example.
#
# The example `using`s Reactant, so training runs through the compiled XLA step (`Flux.trainstep!`);
# epoch 1 therefore includes the one-time step compilation. Comment out `using Reactant` (and
# uncomment `using CUDA`) in the example to bench the eager GPU path instead.

include(joinpath(@__DIR__, "..", "resnet_tinyimagenet.jl"))

using Printf

function bench(; epochs=3, batchsize=128, lr=1e-3, num_workers=4)
    device = isdefined(Main, :Reactant) ? reactant_device() : gpu_device()
    @info "Setup" device epochs batchsize lr num_workers

    train_ds = load_dataset("zh-plus/tiny-imagenet", split="train")
    train_ds = train_ds.cast_column("image", datasets.Image(mode="RGB"))
    train_data = mapobs(train_transform, train_ds)
    train_loader = Flux.DataLoader(train_data; batchsize, shuffle=true, num_workers)
    train_loader = device(train_loader)  # move each batch to the device, free the previous one

    model = resnet18() |> device
    nparams = sum(length, Flux.trainables(model))
    @printf "[model] resnet18  params=%.2fM\n" nparams/1e6
    opt_state = Flux.setup(AdamW(lr), model)

    trainmode!(model)
    for epoch in 1:epochs
        nimg = 0
        t = @elapsed for (x, y) in train_loader
            Flux.trainstep!(loss_fn, model, (x, y), opt_state)
            nimg += length(y)
        end
        @printf "[train] epoch=%d time=%.2fs throughput=%.0f img/s\n" epoch t nimg/t
        flush(stdout)
    end
    return model
end

bench(; epochs=3)

# The included example defines `function (@main)(ARGS)`, which Julia 1.12 auto-invokes after the
# script body — that would launch a full default (30-epoch) training run at exit. Exit explicitly to
# suppress it (this still runs atexit hooks, so Distributed reaps the DataLoader workers cleanly).
exit()
