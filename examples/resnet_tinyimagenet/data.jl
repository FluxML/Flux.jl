# The data pipeline, independent of the framework: flux-cuda/ and flux-reactant/ read it through
# training.jl, and lux-reactant-nitro/ includes it directly.

# Before Flux or Lux: CondaPkg resolves the Python environment when PythonCall loads, capping libstdc++
# at Julia's, which Julia 1.13 fails to detect once CompilerSupportLibraries_jll is loaded
# (JuliaLang/julia#63967). Without the cap, the environment re-resolves whenever the load order changes.
using HuggingFaceDatasets

# ImageNet per-channel mean/std (Tiny-ImageNet is an ImageNet subset), shaped for (W, H, C, N)
# broadcasting.
const NCLASSES = 200
const MEAN = reshape(Float32[0.485, 0.456, 0.406], 1, 1, 3, 1)
const STD  = reshape(Float32[0.229, 0.224, 0.225], 1, 1, 3, 1)

# Decode a raw batch to normalized WHCN Float32. Under HuggingFaceDatasets' "julia" format an
# image column is a stacked (C, W, H, N) UInt8 array — channel axis first, from the numpy->Julia
# axis reversal — so permute to Flux's (W, H, C, N) and standardize per channel.
function decode(batch)
    x = Float32.(batch["image"]) ./ 255f0     # (C, W, H, N)
    x = permutedims(x, (2, 3, 1, 4))          # (W, H, C, N) — Flux WHCN layout
    x = (x .- MEAN) ./ STD
    return (x, batch["label"] .+ 1)           # class labels 1:200 (0-based in the dataset)
end

# Standard small-image training augmentation, applied per image: zero-pad by 4 and take a random
# 64x64 crop, then flip horizontally with probability 1/2.
function augment(xy)
    x, y = xy
    W, H, C, N = size(x)
    pad = 4
    out = similar(x)
    padded = zeros(Float32, W + 2pad, H + 2pad, C)
    for n in 1:N
        @views padded[pad+1:pad+W, pad+1:pad+H, :] .= x[:, :, :, n]
        i, j = rand(0:2pad), rand(0:2pad)                  # random top-left crop offset
        if rand(Bool)
            @views out[:, :, :, n] .= padded[i+W:-1:i+1, j+1:j+H, :]   # crop + horizontal flip
        else
            @views out[:, :, :, n] .= padded[i+1:i+W, j+1:j+H, :]      # crop
        end
    end
    return (out, y)
end

# On-the-fly training pipeline: decode then augment. A named function (not a closure) so the
# `num_workers` path can ship it — and the module globals it reads — to worker processes.
train_transform(batch) = augment(decode(batch))

function tinyimagenet(split)
    ds = load_dataset("zh-plus/tiny-imagenet"; split)
    # A handful of images are grayscale: decode every image as 3-channel RGB so that the batches
    # always stack to a uniform (C, W, H, N) array.
    return ds.cast_column("image", datasets.Image(mode="RGB"))
end
