"""PyTorch port of the Flux GPT-2 example in the parent directory (../train.jl).

Adapted from Andrej Karpathy's build-nanogpt (https://github.com/karpathy/build-nanogpt, MIT
license). It reads the token files written by ../data.jl and uses the same presets,
hyperparameters, and logging as the Flux script, so the two can be compared on the same data.
Differences from build-nanogpt, to match Flux: gradients are clipped per parameter (Flux's
`ClipNorm`), not by their global norm, and `--attn math` selects materialized attention (what
Flux runs under Reactant) instead of flash attention.
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

GPT2_SMALL = dict(vocab_size=50304, block_size=1024, n_layer=12, n_head=12, n_embd=768)
GPT2_TINY = dict(vocab_size=50304, block_size=512, n_layer=6, n_head=6, n_embd=384)
NTOKENS = 50257
EOT = 50256

PRESETS = {
    "fineweb": dict(model=GPT2_SMALL, dataset="fineweb", total_batch=2**19, batchsize=16,
                    lr=6e-4, warmup=715, steps=19073, weight_decay=0.1,
                    eval_every=250, val_steps=20, hellaswag_every=250),
    "tinystories": dict(model=GPT2_TINY, dataset="tinystories", total_batch=2**16, batchsize=32,
                        lr=1e-3, warmup=100, steps=7232, weight_decay=0.1,
                        eval_every=250, val_steps=20, hellaswag_every=0),
}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["n_embd"]
        self.n_head = cfg["n_head"]
        self.c_attn = nn.Linear(d, 3 * d)
        self.c_proj = nn.Linear(d, d)
        self.c_proj.RESIDUAL_SCALE = True

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["n_embd"]
        self.ln_1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.mlp[2].RESIDUAL_SCALE = True

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.wpe = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            if getattr(module, "RESIDUAL_SCALE", False):
                std *= (2 * self.cfg["n_layer"]) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx):
        T = idx.size(1)
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for block in self.h:
            x = block(x)
        return F.linear(self.ln_f(x), self.wte.weight)  # tied output head


def lm_loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))


def learning_rate(step, p):
    if step <= p["warmup"]:
        return p["lr"] * step / p["warmup"]
    progress = min((step - p["warmup"]) / (p["steps"] - p["warmup"]), 1)
    return p["lr"] / 10 + (p["lr"] - p["lr"] / 10) * (1 + math.cos(math.pi * progress)) / 2


class TokenLoader:
    def __init__(self, path, B, T):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.B, self.T = B, T

    def getbatch(self, pos):
        n = self.B * self.T
        if pos + n + 1 > len(self.tokens):
            pos = 0
        buf = torch.from_numpy(self.tokens[pos:pos + n + 1].astype(np.int64))
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        return (x, y), pos + n


@torch.no_grad()
def generate(model, prompt, n=4, length=32, topk=50, device="cuda"):
    x = torch.tensor(prompt, device=device).repeat(n, 1)
    for _ in range(length):
        logits = model(x[:, -model.cfg["block_size"]:])[:, -1, :NTOKENS].float()
        vals, idx = torch.topk(logits, topk, dim=-1)
        choice = torch.multinomial(F.softmax(vals, dim=-1), 1)
        x = torch.cat([x, torch.gather(idx, -1, choice)], dim=1)
    return x.tolist()


def hellaswag_data(tok):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    examples = []
    for ex in ds:
        ctx = tok.encode(ex["ctx"]).ids
        rows = [(ctx, tok.encode(" " + e).ids) for e in ex["endings"]]
        examples.append((rows, int(ex["label"])))
    return examples


@torch.no_grad()
def hellaswag_eval(model, examples, autocast, device="cuda"):
    correct = correct_norm = 0
    for rows, label in examples:
        L = max(len(c) + len(e) for c, e in rows)
        tokens = torch.full((4, L), EOT, dtype=torch.long)
        mask = torch.zeros((4, L))
        for i, (c, e) in enumerate(rows):
            tokens[i, :len(c) + len(e)] = torch.tensor(c + e)
            mask[i, len(c):len(c) + len(e)] = 1
        tokens, mask = tokens.to(device), mask.to(device)
        with autocast:
            logits = model(tokens)
        nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                              tokens[:, 1:].reshape(-1), reduction="none").view(4, -1)
        m = mask[:, 1:]
        sums = (nll * m).sum(1)
        correct += int(sums.argmin() == label)
        correct_norm += int((sums / m.sum(1)).argmin() == label)
    return correct / len(examples), correct_norm / len(examples)


def main():
    ap = argparse.ArgumentParser(description="Pretrain GPT-2 with PyTorch (port of ../train.jl).")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--preset", default="tinystories", choices=PRESETS)
    ap.add_argument("--data-dir", default=os.path.join(here, "..", "data"))
    ap.add_argument("--log-dir", default=os.path.join(here, "log"))
    ap.add_argument("--batchsize", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--hellaswag-every", type=int)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "tf32", "f32"],
                    help="bf16 autocast, or Float32 with TF32 or exact matmuls")
    ap.add_argument("--attn", default="flash", choices=["flash", "math"],
                    help="flash attention, or materialized attention as Flux on Reactant runs it")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peak-tflops", type=float)
    args = ap.parse_args()

    p = dict(PRESETS[args.preset])
    for k in ["batchsize", "steps", "lr", "eval_every", "hellaswag_every"]:
        if getattr(args, k) is not None:
            p[k] = getattr(args, k)
    cfg = p["model"]
    B, T = p["batchsize"], cfg["block_size"]
    assert p["total_batch"] % (B * T) == 0
    accum = p["total_batch"] // (B * T)

    device = "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.precision != "f32"
    torch.backends.cudnn.allow_tf32 = args.precision != "f32"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
    attn_backend = SDPBackend.FLASH_ATTENTION if args.attn == "flash" else SDPBackend.MATH
    if args.precision != "bf16" and args.attn == "flash":
        attn_backend = SDPBackend.EFFICIENT_ATTENTION  # flash attention needs half precision

    data = os.path.join(args.data_dir, p["dataset"])
    train_loader = TokenLoader(os.path.join(data, "train.bin"), B, T)
    val_loader = TokenLoader(os.path.join(data, "val.bin"), B, T)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained("openai-community/gpt2")
    prompt = tok.encode("Hello, I'm a language model,").ids
    hellaswag = hellaswag_data(tok) if p["hellaswag_every"] > 0 else None

    model = GPT(cfg).to(device)
    nparams = sum(q.numel() for q in model.parameters())
    flops = 6 * (nparams - model.wpe.weight.numel()) + 12 * cfg["n_layer"] * cfg["n_embd"] * T
    fwd = torch.compile(model) if args.compile else model
    decay = [q for q in model.parameters() if q.dim() >= 2]
    no_decay = [q for q in model.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": p["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=p["lr"], betas=(0.9, 0.95), eps=1e-8, fused=True)

    os.makedirs(args.log_dir, exist_ok=True)
    logfile = open(os.path.join(args.log_dir, "log.txt"), "a")

    def logmsg(msg):
        print(msg, flush=True)
        print(msg, file=logfile, flush=True)

    logmsg(f"[setup] preset={args.preset} params={nparams / 1e6:.1f}M batch={B}×{T} accum={accum} "
           f"tokens/step={p['total_batch']} precision={args.precision} attn={args.attn} steps={p['steps']}")

    val_batches = [val_loader.getbatch(i * B * T)[0] for i in range(p["val_steps"])]
    pos = 0
    with sdpa_kernel(attn_backend):
        for step in range(1, p["steps"] + 1):
            t0 = time.time()
            lr = learning_rate(step, p)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for _ in range(accum):
                (x, y), pos = train_loader.getbatch(pos)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast:
                    loss = lm_loss(fwd, x, y) / accum
                loss.backward()
                loss_acc += loss.detach()
            for q in model.parameters():
                torch.nn.utils.clip_grad_norm_(q, 1.0)
            opt.step()
            loss_acc = loss_acc.item()  # synchronizes
            dt = time.time() - t0
            tps = p["total_batch"] / dt
            tflops = flops * tps / 1e12
            mfu = f" mfu={100 * tflops / args.peak_tflops:.1f}%" if args.peak_tflops else ""
            logmsg(f"[train] step={step} loss={loss_acc:.4f} lr={lr:.2e} dt={dt:.2f}s "
                   f"tok/s={tps:.0f} tflops={tflops:.1f}{mfu}")

            last = step == p["steps"]
            if p["eval_every"] > 0 and (step % p["eval_every"] == 0 or last):
                model.eval()
                with torch.no_grad(), autocast:
                    vl = sum(lm_loss(fwd, x.to(device), y.to(device)).item() for x, y in val_batches)
                logmsg(f"[eval] step={step} val_loss={vl / len(val_batches):.4f}")
                with autocast:
                    for i, s in enumerate(generate(model, prompt, device=device), 1):
                        logmsg(f"[sample {i}] {tok.decode(s)!r}")
                model.train()
            if hellaswag is not None and (step % p["hellaswag_every"] == 0 or last):
                model.eval()
                acc, acc_norm = hellaswag_eval(model, hellaswag, autocast, device)
                logmsg(f"[hellaswag] step={step} acc={acc:.4f} acc_norm={acc_norm:.4f}")
                model.train()


if __name__ == "__main__":
    main()
