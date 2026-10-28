"""Detection metrics: AUROC, threshold calibration, and bootstrap intervals.

Convention everywhere: a higher score means "more likely seen in training".
"""

import numpy as np
from scipy.stats import rankdata


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 * P(tie), via the Mann-Whitney U statistic."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    ranks = rankdata(np.concatenate([pos, neg]))  # ties get average ranks
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def threshold_at_fpr(neg_calib, alpha=0.05):
    """Smallest threshold t such that the share of calibration negatives with score > t is <= alpha."""
    neg = np.sort(np.asarray(neg_calib, float))
    n = len(neg)
    allowed = int(np.floor(alpha * n))  # negatives we may let through
    return neg[n - allowed - 1]


def rate_above(scores, t):
    return float(np.mean(np.asarray(scores, float) > t))


def bootstrap_auroc(pos, neg, n_boot=2000, seed=0, pos_groups=None, neg_groups=None):
    """Percentile 95% CI for AUROC.

    Resamples positives and negatives separately. If group labels are given (e.g. GSM8K
    template ids), resamples whole groups instead of single items.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)

    def draw(x, groups):
        if groups is None:
            return x[rng.integers(0, len(x), len(x))]
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        picked = rng.choice(uniq, len(uniq), replace=True)
        return np.concatenate([x[groups == g] for g in picked])

    stats = [auroc(draw(pos, pos_groups), draw(neg, neg_groups)) for _ in range(n_boot)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def paired_auroc(a, b):
    """For matched pairs (a_i, b_i): P(a_i > b_i) + 0.5 * P(tie)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(a > b) + 0.5 * np.mean(a == b))


def bootstrap_paired(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        stats.append(paired_auroc(a[idx], b[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)
