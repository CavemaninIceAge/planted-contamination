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
def token_stats(model, tokenizer, texts, device, batch_size=16, max_len=512, max_batch_tokens=4096,
                row_chunk=1024):
    """Score each text with a BOS token prepended, so the first real token is scored too.

    Texts are batched longest first, so padding stays small, and a batch holds at most
    `max_batch_tokens` tokens. The softmax over the vocabulary is taken only at real
    (non-padding) positions, `row_chunk` positions at a time, which keeps memory bounded.
    Results come back in the order of `texts`.
    """
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    encoded = [[bos] + tokenizer(t)["input_ids"][: max_len - 1] for t in texts]
    order = sorted(range(len(encoded)), key=lambda i: -len(encoded[i]))
    out = [None] * len(encoded)
    model.eval()
    start = 0
    while start < len(order):
        width = len(encoded[order[start]])
        chunk = order[start : start + max(1, min(batch_size, max_batch_tokens // width))]
        start += len(chunk)
        ids = torch.full((len(chunk), width), bos, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, i in enumerate(chunk):
            ids[j, : len(encoded[i])] = torch.tensor(encoded[i])
            mask[j, : len(encoded[i])] = 1
        ids, mask = ids.to(device), mask.to(device)
        valid = mask[:, 1:].bool()
        rows = model(input_ids=ids, attention_mask=mask).logits[:, :-1][valid].float()  # (n_valid, vocab)
        target = ids[:, 1:][valid]
        lp, mu, sigma = [], [], []
        for r in range(0, rows.size(0), row_chunk):
            logprobs = torch.log_softmax(rows[r : r + row_chunk], dim=-1)
            probs = logprobs.exp()
            m = (probs * logprobs).sum(-1)
            lp.append(logprobs.gather(-1, target[r : r + row_chunk, None]).squeeze(-1))
            mu.append(m)
            sigma.append((probs * (logprobs - m[:, None]).square()).sum(-1).clamp_min(1e-12).sqrt())
        lp, mu, sigma = (torch.cat(x).cpu().numpy() for x in (lp, mu, sigma))
        pos = 0
        for i, n in zip(chunk, valid.sum(1).tolist()):
            out[i] = TokenStats(lp[pos : pos + n], mu[pos : pos + n], sigma[pos : pos + n])
            pos += n
        del rows
    if device.type == "mps":
        torch.mps.empty_cache()
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
