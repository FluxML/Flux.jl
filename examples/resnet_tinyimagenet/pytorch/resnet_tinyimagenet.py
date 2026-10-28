import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import datasets as hfds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NCLASSES = 200

# ------------------------------------------------------------------------------------------------
# Data
#
# ImageNet per-channel mean/std (Tiny-ImageNet is an ImageNet subset), shaped for (C, H, W)
# broadcasting over an NCHW batch.
MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


def decode(batch):
    """Decode a raw HF batch to a normalized NCHW Float32 tensor + label tensor.

    The dataset yields PIL images; `datasets` with `.with_format("torch")` on an Image column
    hands us a uint8 (N, C, H, W) tensor already in channel-first layout. Move to the device,
    scale to [0,1], standardize per channel (MEAN/STD live on the device).
    """
    x = batch["image"].to(DEVICE, non_blocking=True).to(torch.float32).div_(255.0)  # (N, C, H, W)
    x = (x - MEAN) / STD
    return x, batch["label"].to(DEVICE, non_blocking=True)


def augment(x):
    """Standard small-image augmentation, applied per batch on the GPU: zero-pad by 4 and take a
    random 64x64 crop, then flip horizontally with probability 1/2. Vectorized over the batch —
    every image in the batch shares one crop offset / flip decision, unlike the Julia version
    (which is per-image), but far cheaper."""
    N, C, H, W = x.shape
    pad = 4
    x = F.pad(x, (pad, pad, pad, pad))  # zero-pad H and W
    i = torch.randint(0, 2 * pad + 1, (1,)).item()
    j = torch.randint(0, 2 * pad + 1, (1,)).item()
    x = x[:, :, i:i + H, j:j + W]
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=(3,))
    return x


# ------------------------------------------------------------------------------------------------
# Model: ResNet-18 adapted for small (64x64) images. See the Julia file for the design rationale;
# this is a direct translation.


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return F.relu(self.convs(x) + self.shortcut(x))


def resnet_stage(inplanes, planes, nblocks, stride):
    blocks = [BasicBlock(inplanes, planes, stride=stride)]
    for _ in range(1, nblocks):
        blocks.append(BasicBlock(planes, planes))
    return nn.Sequential(*blocks)


def resnet18(nclasses=NCLASSES):
    return nn.Sequential(
        # small-image stem: 3x3 stride-1, no max-pool (keeps 64x64 resolution)
        nn.Conv2d(3, 64, 3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        resnet_stage(64, 64, 2, stride=1),      # 64x64
        resnet_stage(64, 128, 2, stride=2),     # 32x32
        resnet_stage(128, 256, 2, stride=2),    # 16x16
        resnet_stage(256, 512, 2, stride=2),    #  8x8
        nn.AdaptiveAvgPool2d(1),                # global average pool -> 1x1
        nn.Flatten(),                           # (N, 512)
        nn.Linear(512, nclasses),
    )


# ------------------------------------------------------------------------------------------------
# Training


@torch.no_grad()
def evaluate(loader, model):
    model.eval()
    correct, total, lsum = 0, 0, 0.0
    for batch in loader:
        x, y = decode(batch)
        logits = model(x)
        lsum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return lsum / total, correct / total


def main(epochs=30, batchsize=128, lr=1e-3, weight_decay=0.0, num_workers=4, seed=0,
         clip_norm=False, use_compile=False):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    train_ds = load_dataset("zh-plus/tiny-imagenet", split="train")
    val_ds = load_dataset("zh-plus/tiny-imagenet", split="valid")
    # A handful of Tiny-ImageNet images are grayscale; force RGB so every image stacks uniformly.
    train_ds = train_ds.cast_column("image", hfds.Image(mode="RGB")).with_format("torch")
    val_ds = val_ds.cast_column("image", hfds.Image(mode="RGB")).with_format("torch")

    # As in Flux, the last, smaller training batch of an epoch is dropped.
    train_loader = DataLoader(
        train_ds, batch_size=batchsize, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batchsize, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )

    global MEAN, STD
    MEAN, STD = MEAN.to(DEVICE), STD.to(DEVICE)

    model = resnet18().to(DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[setup] device={DEVICE} params={nparams/1e6:.2f}M epochs={epochs} batchsize={batchsize} "
          f"lr={lr:g} weight_decay={weight_decay:g} num_workers={num_workers} compile={use_compile}")
    # torch.compile (TorchInductor) is PyTorch's analogue of Reactant's XLA compilation: it captures
    # the forward — and, via AOTAutograd, the backward — into fused Triton/cuDNN kernels. The optimiser
    # step and the augmentation stay eager, so this compiles less of the loop than Reactant, which
    # fuses forward + reverse + update into a single executable.
    if use_compile:
        model = torch.compile(model)
    # Flux's AdamW defaults to no weight decay; torch's to 0.01.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def log_eval(epoch):
        val_loss, val_acc = evaluate(val_loader, model)
        print(f"[eval] epoch={epoch} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

    log_eval(0)
    for epoch in range(1, epochs + 1):
        lr_epoch = lr * (1 + math.cos(math.pi * (epoch - 1) / epochs)) / 2   # cosine annealing towards 0
        for g in opt.param_groups:
            g["lr"] = lr_epoch
        model.train()
        lsum = torch.zeros((), device=DEVICE)
        correct = torch.zeros((), dtype=torch.long, device=DEVICE)
        nimg = 0
        t0 = time.perf_counter()
        for batch in train_loader:
            x, y = decode(batch)
            x = augment(x)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            if clip_norm:   # per parameter, as Flux's ClipNorm
                for q in model.parameters():
                    torch.nn.utils.clip_grad_norm_(q, 10.0)
            opt.step()
            lsum += loss.detach() * y.numel()
            correct += (logits.detach().argmax(1) == y).sum()
            nimg += y.numel()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()   # CUDA is async — wait for the queued work before timing
        dt = time.perf_counter() - t0
        print(f"[train] epoch={epoch} lr={lr_epoch:.2e} loss={lsum.item()/nimg:.4f} "
              f"acc={correct.item()/nimg:.4f} time={dt:.1f}s img/s={nimg/dt:.0f}")
        log_eval(epoch)

    return model


def parse_cli():
    p = argparse.ArgumentParser(description="Train a small-image ResNet-18 on Tiny-ImageNet-200.")
    p.add_argument("--epochs", type=int, default=30, help="number of training epochs")
    p.add_argument("--batchsize", type=int, default=128, help="minibatch size")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="peak AdamW learning rate (cosine-annealed towards 0 over training)")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW decoupled weight decay (0 = plain Adam)")
    p.add_argument("--num-workers", type=int, default=4, help="data-loading worker processes")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--clip-norm", action="store_true",
                   help="clip the gradient norm of each parameter to 10")
    p.add_argument("--compile", action="store_true",
                   help="wrap the model in torch.compile (TorchInductor) — PyTorch's XLA analogue")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    main(epochs=args.epochs, batchsize=args.batchsize, lr=args.lr, weight_decay=args.weight_decay,
         num_workers=args.num_workers, seed=args.seed, clip_norm=args.clip_norm,
         use_compile=args.compile)
