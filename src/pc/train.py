"""Continued pretraining on the contaminated corpus: a plain PyTorch loop (no Trainer).

Everything runs in float32 by default. Pythia's logits share a large offset (mean |logit| about
740 on pythia-160m), so under bfloat16 autocast the rounding of the logits alone raised the
cross-entropy of a WikiText batch from 3.81 to 4.36 nats; keeping only the head in float32
still left +0.07. `amp=True` is kept for speed experiments only.
"""

import math
import time

import torch
import torch.nn.functional as F


def lm_loss(model, batch, amp):
    """Next-token cross-entropy over every position of fully packed blocks."""
    with torch.autocast(batch.device.type, dtype=torch.bfloat16, enabled=amp):
        logits = model(input_ids=batch).logits
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), batch[:, 1:].reshape(-1))


def lr_at(step, total, peak, warmup, floor=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def train(model, blocks, device, peak_lr=1e-4, micro_batch=8, accum=2, warmup=30,
          weight_decay=0.1, clip=1.0, seed=0, amp=False):
    torch.manual_seed(seed)
    model.train()
    model.config.use_cache = False
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=peak_lr, betas=(0.9, 0.95), eps=1e-8,
    )
    per_step = micro_batch * accum
    total = len(blocks) // per_step
    log = []
    start = time.time()
    for step in range(total):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, peak_lr, warmup)
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for a in range(accum):
            lo = (step * accum + a) * micro_batch
            batch = torch.from_numpy(blocks[lo : lo + micro_batch]).to(device)
            loss = lm_loss(model, batch, amp) / accum
            loss.backward()
            step_loss += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()
        opt.step()
        log.append({"step": step, "loss": step_loss, "lr": opt.param_groups[0]["lr"], "grad_norm": grad_norm})
        if step % 50 == 0 or step == total - 1:
            rate = (step + 1) * per_step * blocks.shape[1] / (time.time() - start)
            print(f"step {step:4d}/{total}  loss {step_loss:.4f}  lr {log[-1]['lr']:.2e}  {rate:,.0f} tok/s", flush=True)
    model.eval()
    return log
