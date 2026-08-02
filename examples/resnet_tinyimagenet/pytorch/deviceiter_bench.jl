# Does wrapping the DataLoader in an MLDataDevices device iterator (`DEVICE(loader)`) help?
#
# Two training paths, same model/data/optimizer, measured back-to-back in one process:
#   baseline    — batch stays on CPU; the step moves `x |> DEVICE` and onehots `y` on CPU.
#   deviceiter  — `DEVICE(loader)` yields GPU batches; onehot is done on the GPU. The device
#                 iterator `unsafe_free!`s each previous batch, so per-step input arrays don't
#                 pile up in the pool.
#
# For each path we report steady-state per-epoch train time AND the peak CUDA memory-pool
# reservation (`cached`) and live usage (`used`). Run:
#   julia --project=. -t auto pytorch/deviceiter_bench.jl

include(joinpath(@__DIR__, "..", "resnet_tinyimagenet.jl"))

using Printf

const MB = 1024^2

peak_mem() = (CUDA.cached_memory() / MB, CUDA.used_memory() / MB)  # (reserved, live) in MiB

function make_data(; batchsize, num_workers)
    train_ds = load_dataset("zh-plus/tiny-imagenet", split="train")
    train_ds = train_ds.cast_column("image", datasets.Image(mode="RGB"))
    train_data = mapobs(train_transform, train_ds)
    return Flux.DataLoader(train_data; batchsize, shuffle=true, num_workers)
end

# One training epoch, returning (seconds, peak_reserved_MiB, peak_live_MiB).
function train_epoch!(model, opt, data; on_device::Bool)
    nimg = Ref(0)
    creserved = Ref(0.0); clive = Ref(0.0)
    function note!()
        r, l = peak_mem()
        creserved[] = max(creserved[], r); clive[] = max(clive[], l)
    end
    t0 = time()
    if on_device
        Flux.train!(model, data, opt) do m, x, y            # x, y already on the GPU
            nimg[] += length(y)
            logitcrossentropy(m(x), onehotbatch(y, 0:NCLASSES-1))
        end
    else
        Flux.train!(model, data, opt) do m, x, y            # x, y on the CPU
            nimg[] += length(y)
            logitcrossentropy(m(x |> DEVICE), onehotbatch(y, 0:NCLASSES-1) |> DEVICE)
        end
    end
    CUDA.synchronize(); note!()
    return time() - t0, nimg[], creserved[], clive[]
end

function run_mode(name, model, opt, loader; epochs, on_device)
    @printf "\n=== mode=%s (on_device=%s) ===\n" name on_device
    flush(stdout)
    data = on_device ? DEVICE(loader) : loader
    for epoch in 1:epochs
        trainmode!(model)
        dt, nimg, res, live = train_epoch!(model, opt, data; on_device)
        @printf "[%s] epoch=%d time=%.2fs throughput=%.0f img/s  reserved=%.0f MiB  live=%.0f MiB\n" name epoch dt nimg/dt res live
        flush(stdout)
    end
end

function main(; epochs=3, batchsize=128, lr=1e-3, num_workers=4)
    @info "Setup" DEVICE epochs batchsize lr num_workers
    loader = make_data(; batchsize, num_workers)

    # Fresh model/opt per mode so weights don't carry over; reclaim the pool between modes so each
    # mode's peak reservation is measured from a clean-ish baseline (model+opt state only).
    CUDA.reclaim()
    model1 = resnet18() |> DEVICE
    opt1 = Flux.setup(AdamW(lr), model1)
    run_mode("baseline", model1, opt1, loader; epochs, on_device=false)
    model1 = nothing; opt1 = nothing
    GC.gc(true); CUDA.reclaim()
    let (r, l) = peak_mem()
        @printf "[reclaim] after baseline: reserved=%.0f MiB live=%.0f MiB\n" r l
    end
    flush(stdout)

    model2 = resnet18() |> DEVICE
    opt2 = Flux.setup(AdamW(lr), model2)
    run_mode("deviceiter", model2, opt2, loader; epochs, on_device=true)
end

main(; epochs=3)
