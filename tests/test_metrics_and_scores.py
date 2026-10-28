import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pc import data, metrics, score  # noqa: E402


def test_auroc_matches_mann_whitney():
    rng = np.random.default_rng(0)
    pos = rng.normal(0.5, 1, 200).round(1)  # rounding creates ties
    neg = rng.normal(0.0, 1, 300).round(1)
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    assert np.isclose(metrics.auroc(pos, neg), u / (len(pos) * len(neg)))


def test_auroc_extremes():
    assert metrics.auroc([2, 3], [0, 1]) == 1.0
    assert metrics.auroc([0, 1], [2, 3]) == 0.0
    assert metrics.auroc([1, 1], [1, 1]) == 0.5


def test_conformal_threshold_order_statistic():
    ref = np.arange(1, 101, dtype=float)  # n = 100: ceil(101 * 0.95) = 96th smallest
    assert metrics.conformal_threshold(ref, 0.05) == 96.0
    assert metrics.conformal_threshold(np.arange(10.0), 0.05) == np.inf  # too few to reach 95%


def test_conformal_threshold_false_positive_rate():
    """With exchangeable reference and test draws, the flag rate stays at or below alpha."""
    rng = np.random.default_rng(1)
    for n in (20, 99, 435):
        flagged = [rng.normal() > metrics.conformal_threshold(rng.normal(size=n), 0.05) for _ in range(20000)]
        assert np.mean(flagged) <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 20000)


def test_holm():
    assert np.allclose(metrics.holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])


def test_paired_auroc():
    assert metrics.paired_auroc([1, 2, 3], [0, 2, 4]) == 0.5


class TinyLM(torch.nn.Module):
    """Fixed random bigram model, enough to check the scoring maths."""

    def __init__(self, vocab=11):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.table = torch.randn(vocab, vocab, generator=g)

    def forward(self, input_ids, attention_mask=None):
        class Out:
            pass

        out = Out()
        out.logits = self.table[input_ids]
        return out


class TinyTok:
    bos_token_id = 0
    eos_token_id = 0

    def __call__(self, text):
        return {"input_ids": [1 + (ord(c) % 10) for c in text]}


def test_token_stats_against_brute_force():
    model, tok = TinyLM(), TinyTok()
    texts = ["abcde", "xy", "hello world"]
    stats = score.token_stats(model, tok, texts, torch.device("cpu"), batch_size=2)
    for text, s in zip(texts, stats):
        ids = [0] + tok(text)["input_ids"]
        assert len(s.logp) == len(ids) - 1
        for t in range(1, len(ids)):
            lp = torch.log_softmax(model.table[ids[t - 1]], -1)
            p = lp.exp()
            mu = (p * lp).sum()
            sigma = ((p * (lp - mu) ** 2).sum()).sqrt()
            assert np.isclose(s.logp[t - 1], lp[ids[t]].item(), atol=1e-5)
            assert np.isclose(s.mu[t - 1], mu.item(), atol=1e-5)
            assert np.isclose(s.sigma[t - 1], sigma.item(), atol=1e-5)


def test_detectors_direction_and_mink():
    stats = [score.TokenStats(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]), np.zeros(5), np.ones(5))]
    s = score.detector_scores(["abcde"], stats, ref_nll=np.array([3.0]), k=0.2)
    assert np.isclose(s["loss"][0], -3.0)
    assert np.isclose(s["mink"][0], -5.0)  # lowest 20% of 5 tokens = 1 token
    assert np.isclose(s["minkpp"][0], -5.0)
    assert np.isclose(s["ref"][0], 0.0)


def test_format_item_strips_calculator():
    text = data.format_item("How many?", "2+2 = <<2+2=4>>4 apples.\n#### 4")
    assert "<<" not in text and text.endswith("#### 4")


def test_groups_disjoint_and_complete():
    templates = list(range(0, 1300, 13))[:100]
    g = data.assign_groups(1319, templates, [1, 2, 4, 8, 16], 120, 50, 16, seed=3)
    all_idx = [i for v in g.values() for i in v]
    assert len(all_idx) == len(set(all_idx)) == 1319
    assert all(len(g[f"k{k}"]) == 120 for k in [1, 2, 4, 8, 16])
    assert sorted(g["tmpl_k16"] + g["tmpl_unseen"]) == templates and len(g["tmpl_k16"]) == 50
    assert not set(templates) & set(g["unseen"] + g["k1"] + g["k16"])
    counts = data.planted_counts(g)
    assert len(counts) == 650 and counts[g["tmpl_k16"][0]] == 16 and counts[g["k2"][0]] == 2


class CharTok:
    eos_token_id = 0

    def __call__(self, text):
        return {"input_ids": [ord(c) for c in text]}


def test_corpus_has_exact_copy_counts():
    tok = CharTok()
    items = [f"Question: item {i} ### Answer: {i * 7}" for i in range(30)]
    counts = {3: 1, 5: 4, 8: 16}
    wiki = [("lorem ipsum dolor sit amet " * 40) + str(i) for i in range(400)]
    blocks, where, stats = data.build_blocks(tok, wiki, items, counts, 120_000, 512, seed=0)
    stream = "".join(chr(t) if t else "|" for t in blocks.reshape(-1))
    for i, item in enumerate(items):
        assert stream.count("|" + item + "|") == counts.get(i, 0)
    assert {i: len(v) for i, v in where.items()} == counts
    assert all(len(set(v)) == len(v) for v in where.values())  # copies sit in different blocks


def test_block_count_is_multiple_of_step():
    tok = CharTok()
    items = [f"Question: item {i} ### Answer: {i * 7}" for i in range(30)]
    wiki = [("lorem ipsum dolor sit amet " * 40) + str(i) for i in range(400)]
    counts = {3: 1, 5: 4, 8: 16}
    blocks, where, _ = data.build_blocks(tok, wiki, items, counts, 120_000, 512, seed=0, multiple=16)
    assert len(blocks) % 16 == 0
    assert {i: len(v) for i, v in where.items()} == counts


def test_token_stats_order_does_not_depend_on_batching():
    model, tok = TinyLM(), TinyTok()
    texts = ["a", "abcdefgh", "abc", "hello world", "xy"]
    one = score.token_stats(model, tok, texts, torch.device("cpu"), batch_size=1)
    many = score.token_stats(model, tok, texts, torch.device("cpu"), batch_size=4, max_batch_tokens=20, row_chunk=3)
    for a, b in zip(one, many):
        assert np.allclose(a.logp, b.logp, atol=1e-6) and np.allclose(a.sigma, b.sigma, atol=1e-6)


def test_surface_features():
    f = data.surface_features("Costs €14.0 and 2,500 \u00f7 5")
    assert f["other_currency"] == 1 and f["dot_zero"] == 1 and f["longest_number"] == 3


def test_normalize_surface():
    t = data.normalize_surface("Question: It costs €14.0 and ₣3.\nAnswer: 12 ÷ 4 = 3\n\n#### 3".replace("\\n", "\n"))
    assert "€" not in t and "$14 " in t and "$3." in t and "/" in t and "\n\n####" not in t


def test_padding_does_not_change_scores():
    """Batched (right-padded) scoring must match one-by-one scoring on a real Pythia model."""
    import pytest

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = "EleutherAI/pythia-70m-deduped"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    except Exception:
        pytest.skip("pythia-70m-deduped not available offline")
    texts = ["Question: short one?\nAnswer: 4\n#### 4",
             "Question: " + "a much longer question with many words " * 6 + "\nAnswer: 12\n#### 12"]
    batched = score.token_stats(model, tok, texts, torch.device("cpu"), batch_size=2)
    single = [score.token_stats(model, tok, [t], torch.device("cpu"), batch_size=1)[0] for t in texts]
    # padded batches take a different attention kernel path, so allow float32 round-off only
    for b, s in zip(batched, single):
        assert len(b.logp) == len(s.logp)
        assert np.abs(b.logp - s.logp).max() < 5e-3 and np.abs(b.logp - s.logp).mean() < 1e-3
        assert np.abs(b.mu - s.mu).max() < 5e-3 and np.abs(b.sigma - s.sigma).max() < 5e-3
