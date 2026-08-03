# PyTorch counterpart of benchmark.jl: measures the speedup and memory effect of
# `torch.autocast` (AMP) on the SAME small-image ResNet-18, batch sizes, and GPU, so the
# Flux `autocast` numbers can be compared against PyTorch's native mixed precision.
#
# Same methodology as the Julia script: median wall-clock time with a CUDA sync, and the
# exact peak of live allocated bytes (`torch.cuda.max_memory_allocated`) for a fwd+bwd step.
# Like the Flux benchmark, the backward runs without a GradScaler, so the timing reflects
# the autocast casts alone (GradScaler affects fp16 numerics, not speed).
#
# Run with uv:
#
#     uv run --python 3.12 --with 'torch --index-url https://download.pytorch.org/whl/cu128' \
#         perf/autocast/benchmark_pytorch.py
#
# (or activate an env that has a Blackwell-capable torch, i.e. torch >= 2.7 with cu128, and
#  run `python perf/autocast/benchmark_pytorch.py`).

import statistics
import time

import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NCLASSES = 200

# Fixed input sizes ⇒ let cuDNN autotune its convolution algorithms (standard training
# practice; the Flux/NNlib path likewise caches a chosen algorithm per shape).
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------------------
# Small-image ResNet-18 (matches resnet18() in benchmark.jl: 3x3 stride-1 stem, no initial
# maxpool, 4 stages [64,128,256,512] x 2 blocks, strides [1,2,2,2]).
# ---------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


def resnet_stage(inplanes, planes, nblocks, stride):
    blocks = [BasicBlock(inplanes, planes, stride)]
    for _ in range(1, nblocks):
        blocks.append(BasicBlock(planes, planes))
    return blocks


def resnet18(nclasses=NCLASSES):
    layers = [nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU()]
    layers += resnet_stage(64, 64, 2, 1)
    layers += resnet_stage(64, 128, 2, 2)
    layers += resnet_stage(128, 256, 2, 2)
    layers += resnet_stage(256, 512, 2, 2)
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, nclasses)]
    return nn.Sequential(*layers)


def make_batch(bs):
    x = torch.randn(bs, 3, 64, 64, device=DEV)
    y = torch.randint(0, NCLASSES, (bs,), device=DEV)
    return x, y


# ---------------------------------------------------------------------------------------
# Timing + peak-memory helpers (mirror the Julia versions)
# ---------------------------------------------------------------------------------------
def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def timed(f, samples=30, warmup=5):
    for _ in range(warmup):
        f()
        sync()
    ts = []
    for _ in range(samples):
        sync()
        t0 = time.perf_counter()
        f()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def peak_used(f):
    if DEV != "cuda":
        f()
        return 0
    f()
    sync()
    torch.cuda.reset_peak_memory_stats()
    sync()
    f()
    sync()
    return torch.cuda.max_memory_allocated()


def fmt_ms(t):
    return f"{1e3 * t:.2f} ms"


def fmt_bytes(n):
    return f"{n / 2**30:.3f} GiB" if DEV == "cuda" else "n/a"


# ---------------------------------------------------------------------------------------
# Speedup + memory: Float32 baseline vs autocast(float16) / autocast(bfloat16)
# ---------------------------------------------------------------------------------------
PRECISIONS = [("Float32 (baseline)", None), ("Float16", torch.float16), ("BFloat16", torch.bfloat16)]


def speedup(bs):
    print("\n" + "─" * 80)
    print(f"● ResNet-18 AMP autocast speed & memory — batch {bs}")
    print("─" * 80)
    print(f"  {'precision':<20} {'forward':>13} {'fwd+bwd':>15} {'peak used (bwd)':>15}")

    crit = nn.CrossEntropyLoss()
    base_fwd = base_step = 0.0
    for label, T in PRECISIONS:
        model = resnet18().to(DEV)
        x, y = make_batch(bs)

        def forward():
            if T is None:
                return model(x)
            with torch.autocast(device_type=DEV, dtype=T):
                return model(x)

        def step():
            model.zero_grad(set_to_none=True)
            if T is None:
                loss = crit(model(x), y)
            else:
                with torch.autocast(device_type=DEV, dtype=T):
                    loss = crit(model(x), y)
            loss.backward()

        t_fwd = timed(forward)
        t_step = timed(step, samples=20)
        mem = peak_used(step)

        if T is None:
            base_fwd, base_step = t_fwd, t_step
            print(f"  {label:<20} {fmt_ms(t_fwd):>13} {fmt_ms(t_step):>15} {fmt_bytes(mem):>15}")
        else:
            print(f"  {label:<20} {fmt_ms(t_fwd):>13} {fmt_ms(t_step):>15} {fmt_bytes(mem):>15}"
                  f"   ({base_fwd / t_fwd:.2f}x / {base_step / t_step:.2f}x vs baseline)")


def main():
    if DEV == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"[info] torch {torch.__version__} on {p.name}, "
              f"{p.total_memory / 2**30:.1f} GiB, cudnn.benchmark={torch.backends.cudnn.benchmark}")
    else:
        print("[info] No CUDA GPU found: running on CPU (only timings are meaningful).")

    for bs in (64, 128):
        speedup(bs)
    print("\nDone.")


if __name__ == "__main__":
    main()
