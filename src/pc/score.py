"""Per-token statistics from a causal LM and the membership scores built on them.

Every detector returns a score where higher means "more likely seen in training".

  loss      -mean token NLL                                   (Yeom et al., 2018)
  zlib      -mean token NLL / zlib-compressed length in bytes (Carlini et al., 2021)
  mink      mean of the lowest k% token log-probs             (Shi et al., 2024, Min-K% Prob)
  minkpp    mean of the lowest k% of (log p - mu) / sigma     (Zhang et al., 2025, Min-K%++)
  ref       NLL_reference - NLL_target                        (difficulty calibration; MIMIR)
"""

import zlib
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TokenStats:
    logp: np.ndarray   # log p(x_t | x_<t) for every scored token
    mu: np.ndarray     # E_{z ~ p(.|x_<t)}[log p(z)]
    sigma: np.ndarray  # std of log p(z) under p(.|x_<t)


@torch.no_grad()
def token_stats(model, tokenizer, texts, device, batch_size=16, max_len=512):
    """Score each text with a BOS token prepended, so the first real token is scored too."""
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    encoded = [[bos] + tokenizer(t)["input_ids"][: max_len - 1] for t in texts]
    out = []
    model.eval()
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        width = max(len(ids) for ids in chunk)
        ids = torch.full((len(chunk), width), bos, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, seq in enumerate(chunk):
            ids[j, : len(seq)] = torch.tensor(seq)
            mask[j, : len(seq)] = 1
        ids, mask = ids.to(device), mask.to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        target = ids[:, 1:]
        lp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        probs = logprobs.exp()
        mu = (probs * logprobs).sum(-1)
        var = (probs * logprobs.pow(2)).sum(-1) - mu.pow(2)
        sigma = var.clamp_min(1e-12).sqrt()
        valid = mask[:, 1:].bool()
        for j in range(len(chunk)):
            v = valid[j]
            out.append(TokenStats(lp[j][v].cpu().numpy(), mu[j][v].cpu().numpy(), sigma[j][v].cpu().numpy()))
    return out


def _lowest_mean(x, frac):
    k = max(1, int(np.ceil(frac * len(x))))
    return float(np.sort(x)[:k].mean())


def mean_nll(stats):
    return np.array([-s.logp.mean() for s in stats])


def detector_scores(texts, target_stats, ref_nll=None, k=0.2):
    """All detector scores for one model. ref_nll (mean NLL of the same texts under a
    reference model) enables `ref`, the log-likelihood difference used by MIMIR."""
    nll = mean_nll(target_stats)
    zlen = np.array([len(zlib.compress(t.encode("utf-8"))) for t in texts])
    scores = {
        "loss": -nll,
        "zlib": -nll / zlen,
        "mink": np.array([_lowest_mean(s.logp, k) for s in target_stats]),
        "minkpp": np.array([_lowest_mean((s.logp - s.mu) / s.sigma, k) for s in target_stats]),
    }
    if ref_nll is not None:
        scores["ref"] = np.asarray(ref_nll) - nll
    return scores
