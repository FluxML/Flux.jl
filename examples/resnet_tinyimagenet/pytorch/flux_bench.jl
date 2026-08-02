# Timed driver for the Flux ResNet-18 / Tiny-ImageNet example, mirroring the PyTorch
# `--benchmark-epochs` run: eval at epoch 0, then N timed train epochs each followed by eval.
# Reuses the model / data / loss definitions from the example file itself.

include(joinpath(@__DIR__, "..", "resnet_tinyimagenet.jl"))

using Printf

function bench(; epochs=3, batchsize=128, lr=1e-3, num_workers=4)
    @info "Setup" DEVICE epochs batchsize lr num_workers

    train_ds = load_dataset("zh-plus/tiny-imagenet", split="train")
    val_ds   = load_dataset("zh-plus/tiny-imagenet", split="valid")
    train_ds = train_ds.cast_column("image", datasets.Image(mode="RGB"))
    val_ds   = val_ds.cast_column("image", datasets.Image(mode="RGB"))

    train_data = mapobs(train_transform, train_ds)
    val_data   = mapobs(decode, val_ds)

    train_loader = Flux.DataLoader(train_data; batchsize, shuffle=true, num_workers)
    val_loader   = Flux.DataLoader(val_data; batchsize, num_workers)

    model = resnet18() |> DEVICE
    nparams = sum(length, Flux.trainables(model))
    @printf "[model] resnet18  params=%.2fM\n" nparams/1e6
    opt = Flux.setup(AdamW(lr), model)

    function run_eval(epoch)
        te = time()
        tl, ta = loss_and_accuracy(train_loader, model)
        vl, va = loss_and_accuracy(val_loader, model)
        @printf "[eval] epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f eval_time=%.1fs\n" epoch tl ta vl va (time()-te)
        flush(stdout)
    end

    run_eval(0)
    for epoch in 1:epochs
        trainmode!(model)
        nimg = Ref(0)
        t0 = time()
        Flux.train!(model, DEVICE(train_loader), opt) do m, x, y   # device iterator: GPU batches
            nimg[] += length(y)
            logitcrossentropy(m(x), onehotbatch(y, 0:NCLASSES-1))
        end
        CUDA.synchronize()
        dt = time() - t0
        @printf "[train] epoch=%d time=%.2fs throughput=%.0f img/s\n" epoch dt nimg[]/dt
        flush(stdout)
        run_eval(epoch)
    end
    return model
end

bench(; epochs=3)

# The included example defines `function (@main)(ARGS)`, which Julia 1.12 auto-invokes after the
# script body — that would launch a full default (30-epoch) training run at exit. Exit explicitly to
# suppress it (this still runs atexit hooks, so Distributed reaps the DataLoader workers cleanly).
exit()
