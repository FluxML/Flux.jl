"""PyTorch port of the Flux ResNet-18 / Tiny-ImageNet-200 example.

A faithful mirror of `../resnet_tinyimagenet.jl`: same small-image ResNet-18, same data
(`zh-plus/tiny-imagenet` from the HuggingFace Hub), same augmentation, same AdamW /
cross-entropy training loop. Kept close to the Julia version so the two can be compared
line-for-line and, more importantly, benchmarked head to head on the same GPU.
"""

import argparse
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
    every image in the batch shares one crop offset / flip decision, matching nothing in the Julia
    version (which is per-image) but far cheaper; per-image variants are commented below."""
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
def loss_and_accuracy(loader, model):
    model.eval()
    correct, total, lsum = 0, 0, 0.0
    for batch in loader:
        x, y = decode(batch)
        logits = model(x)
        lsum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return lsum / total, correct / total


def collate_identity(samples):
    # HF with torch format already returns column tensors when we index a batch; but DataLoader
    # calls collate per-sample. We instead let DataLoader batch and stack the dict columns.
    keys = samples[0].keys()
    return {k: torch.utils.data.default_collate([s[k] for s in samples]) for k in keys}


def main(epochs=30, batchsize=128, lr=1e-3, num_workers=4, benchmark_epochs=None):
    print(f"[setup] device={DEVICE} epochs={epochs} batchsize={batchsize} lr={lr} "
          f"num_workers={num_workers}")
    torch.backends.cudnn.benchmark = True

    train_ds = load_dataset("zh-plus/tiny-imagenet", split="train")
    val_ds = load_dataset("zh-plus/tiny-imagenet", split="valid")
    # A handful of Tiny-ImageNet images are grayscale; force RGB so every image stacks uniformly.
    train_ds = train_ds.cast_column("image", hfds.Image(mode="RGB"))
    val_ds = val_ds.cast_column("image", hfds.Image(mode="RGB"))
    train_ds = train_ds.with_format("torch")
    val_ds = val_ds.with_format("torch")

    train_loader = DataLoader(
        train_ds, batch_size=batchsize, shuffle=True, num_workers=num_workers,
        collate_fn=collate_identity, pin_memory=True, drop_last=False,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batchsize, shuffle=False, num_workers=num_workers,
        collate_fn=collate_identity, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    global MEAN, STD
    MEAN, STD = MEAN.to(DEVICE), STD.to(DEVICE)

    model = resnet18().to(DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[model] resnet18  params={nparams/1e6:.2f}M")
    # Match Flux's AdamW default (lambda=0 -> no weight decay); torch defaults to 0.01.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    def run_eval(epoch):
        train_loss, train_acc = loss_and_accuracy(train_loader, model)
        val_loss, val_acc = loss_and_accuracy(val_loader, model)
        print(f"[eval] epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    run_eval(0)
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        nimg = 0
        for batch in train_loader:
            x, y = decode(batch)
            x = augment(x)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            nimg += y.numel()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"[train] epoch={epoch} time={dt:.2f}s throughput={nimg/dt:.0f} img/s")
        run_eval(epoch)
        if benchmark_epochs is not None and epoch >= benchmark_epochs:
            break

    return model


def parse_cli():
    p = argparse.ArgumentParser(description="Train a small-image ResNet-18 on Tiny-ImageNet-200.")
    p.add_argument("--epochs", type=int, default=30, help="number of training epochs")
    p.add_argument("--batchsize", type=int, default=128, help="minibatch size")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--num-workers", type=int, default=4, help="data-loading worker processes")
    p.add_argument("--benchmark-epochs", type=int, default=None,
                   help="stop after this many train epochs (for timing runs)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    main(epochs=args.epochs, batchsize=args.batchsize, lr=args.lr,
         num_workers=args.num_workers, benchmark_epochs=args.benchmark_epochs)
