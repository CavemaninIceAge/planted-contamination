"""End-to-end run of scripts/analyze.py on synthetic scores with the same layout as the real ones."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import analyze  # noqa: E402
from pc import data  # noqa: E402

N_TEST, N_TRAIN, N_REWRITES = 1319, 1000, 50
TEMPLATES = np.arange(0, 1300, 13)[:100]
SCORES = ["loss", "zlib", "mink", "minkpp", "ref", "nll"]


def frame(rng, with_questions):
    parts = []

    def add(name, idx, template=-1, instance=-1):
        parts.append(pd.DataFrame({"set": name, "idx": idx, "template": template, "instance": instance}))

    add("test", np.arange(N_TEST))
    add("train", np.sort(rng.choice(7473, N_TRAIN, replace=False)))
    rep = np.repeat(TEMPLATES, N_REWRITES)
    inst = np.tile(np.arange(N_REWRITES), len(TEMPLATES))
    for prefix in ([""] + (["q_"] if with_questions else [])):
        if prefix:
            add("q_test", np.arange(N_TEST))
        add(prefix + "sym_raw", rep, rep + 5000, inst)
        add(prefix + "sym_norm", rep, rep + 5000, inst)
        add(prefix + "orig_norm", TEMPLATES)
    df = pd.concat(parts, ignore_index=True)
    for col in SCORES:
        df[col] = rng.normal(size=len(df))
    df["n_tokens"] = rng.integers(60, 300, len(df))
    for col in data.SURFACE_FEATURES:
        df[col] = rng.random(len(df))
    return df


def test_analyze_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "N_PERM", 200)
    monkeypatch.setattr(analyze, "N_BOOT", 100)
    rng = np.random.default_rng(0)
    res = tmp_path / "results"
    res.mkdir()
    base = frame(rng, with_questions=True)
    base.to_csv(res / "scores_base.csv.gz", index=False)
    for seed in (0, 1):
        groups = data.assign_groups(N_TEST, TEMPLATES.tolist(), [1, 2, 4, 8, 16], 120, 50, 16, seed)
        counts = data.planted_counts(groups)
        df = frame(rng, with_questions=False)
        label = {i: g for g, idx in groups.items() for i in idx}
        df["group"] = [label.get(i, "-") if s in ("test", "orig_norm", "sym_raw", "sym_norm") else "-"
                       for s, i in zip(df["set"], df["idx"])]
        bump = df["idx"].map(counts).fillna(0).to_numpy() * (df["set"] == "test")
        for col in ["loss", "zlib", "mink", "minkpp", "ref"]:
            df[col] += 0.2 * bump
        df["oracle"] = 0.2 * bump + rng.normal(size=len(df))
        df["seed"] = seed
        df.to_csv(res / f"scores_seed{seed}.csv.gz", index=False)
        steps = {i: sorted(rng.choice(600, k, replace=False).tolist()) for i, k in counts.items()}
        (res / f"groups_seed{seed}.json").write_text(json.dumps(
            {"groups": groups, "exposure_steps": steps, "total_steps": 600}))
    monkeypatch.setattr(sys, "argv", ["analyze.py", "--results", str(res), "--figures", str(tmp_path / "fig"),
                                      "--tag", "smoke"])
    analyze.main()
    summary = json.loads((res / "summary.json").read_text())
    # planted items were given higher scores: the test bed must see it, the clean audit must not
    assert summary["test_bed"]["loss"]["k16"]["mean"] > 0.95
    assert 0.4 < summary["clean_audit"]["loss"]["raw"]["mean_rank_fraction"] < 0.6
    assert set(summary["hypotheses"]) == {"H1", "H2", "H3"}
    for name in ("threshold_source", "clean_audit", "dose_response"):
        assert (tmp_path / "fig" / f"{name}_smoke.png").exists()
