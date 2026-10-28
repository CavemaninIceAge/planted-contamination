"""Turn per-item scores in results/<tag>/ into summary tables (JSON + Markdown) and figures."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pc import metrics  # noqa: E402

DETECTORS = ["loss", "zlib", "mink", "minkpp", "ref"]
NAMES = {"loss": "Loss", "zlib": "Zlib", "mink": "Min-K%", "minkpp": "Min-K%++", "ref": "Ref (70m)",
         "oracle": "Oracle ref"}
ALPHA = 0.05


def exp1(res):
    seeds = sorted(res.glob("scores_seed*.csv"))
    rows = []
    for path in seeds:
        df = pd.read_csv(path)
        seed = int(df["seed"].iloc[0])
        g = df[df["source"] == "gsm8k"]
        exposures = sorted(int(x[1:]) for x in g["group"].unique() if x.startswith("k"))
        for d in DETECTORS + ["oracle"]:
            calib = g.loc[g["group"] == "calib", d].to_numpy()
            lock = g.loc[g["group"] == "lockbox", d].to_numpy()
            t = metrics.threshold_at_fpr(calib, ALPHA)
            fp = int((lock > t).sum())
            row = {"seed": seed, "detector": d, "fpr_lockbox": fp / len(lock),
                   "fpr_lockbox_ci": metrics.wilson(fp, len(lock)),
                   "auroc_calib_vs_lockbox": metrics.auroc(calib, lock)}
            for k in exposures:
                pos = g.loc[g["group"] == f"k{k}", d].to_numpy()
                row[f"auroc_k{k}"] = metrics.auroc(pos, lock)
                row[f"auroc_k{k}_ci"] = metrics.bootstrap_auroc(pos, lock, seed=seed)
                row[f"tpr_k{k}"] = metrics.rate_above(pos, t)
            rows.append(row)
    return pd.DataFrame(rows), exposures


def exp2(res):
    """Base model: unseen GSM8K originals vs their unseen GSM-Symbolic rewrites."""
    df = pd.read_csv(res / "scores_base.csv")
    sym = df[df["source"] == "symbolic"]
    orig_ids = sorted(sym["item"].unique())
    orig = df[(df["source"] == "gsm8k") & (df["item"].isin(orig_ids))].set_index("item").loc[orig_ids]
    all_gsm = df[df["source"] == "gsm8k"]
    rows = []
    for d in DETECTORS:
        # within-template comparison: each original against its own 50 rewrites
        per_t = [metrics.auroc([orig.loc[i, d]], sym.loc[sym["item"] == i, d]) for i in orig_ids]
        boot = np.random.default_rng(0)
        within_ci = np.percentile([np.mean(boot.choice(per_t, len(per_t))) for _ in range(2000)], [2.5, 97.5])
        # audit simulation: rewrites stand in for "certainly unseen" when picking the threshold
        t = metrics.threshold_at_fpr(sym[d].to_numpy(), ALPHA)
        flagged = int((orig[d] > t).sum())
        flagged_all = int((all_gsm[d] > t).sum())
        rows.append({
            "detector": d,
            "within_template_auroc": float(np.mean(per_t)),
            "within_template_auroc_ci": [float(x) for x in within_ci],
            "auroc_orig_vs_rewrites": metrics.auroc(orig[d], sym[d]),
            "auroc_orig_vs_rewrites_ci": metrics.bootstrap_auroc(orig[d].to_numpy(), sym[d].to_numpy(),
                                                                 neg_groups=sym["item"].to_numpy()),
            "flagged_originals": flagged / len(orig),
            "flagged_originals_ci": metrics.wilson(flagged, len(orig)),
            "flagged_all_gsm8k": flagged_all / len(all_gsm),
            "flagged_all_gsm8k_ci": metrics.wilson(flagged_all, len(all_gsm)),
            "mean_tokens_orig": float(orig["n_tokens"].mean()),
            "mean_tokens_rewrites": float(sym["n_tokens"].mean()),
        })
    return pd.DataFrame(rows)


def spillover(res):
    """Fine-tuned models: do rewrites of planted originals look 'seen'?"""
    rows = []
    for path in sorted(res.glob("scores_seed*.csv")):
        df = pd.read_csv(path)
        sym = df[df["source"] == "symbolic"]
        heavy = sym[sym["group"].isin(["sym_of_k8", "sym_of_k16"])]
        clean = sym[sym["group"].isin(["sym_of_calib", "sym_of_lockbox"])]
        if heavy.empty or clean.empty:
            continue
        for d in DETECTORS + ["oracle"]:
            rows.append({"seed": int(df["seed"].iloc[0]), "detector": d,
                         "n_templates_heavy": int(heavy["item"].nunique()),
                         "n_templates_clean": int(clean["item"].nunique()),
                         "auroc": metrics.auroc(heavy[d], clean[d]),
                         "auroc_ci": metrics.bootstrap_auroc(heavy[d].to_numpy(), clean[d].to_numpy(),
                                                             pos_groups=heavy["item"].to_numpy(),
                                                             neg_groups=clean["item"].to_numpy())})
    return pd.DataFrame(rows)


def fmt(x):
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pythia-160m-deduped")
    args = ap.parse_args()
    res = ROOT / "results" / args.tag
    fig = ROOT / "figures"
    fig.mkdir(exist_ok=True)

    e1, exposures = exp1(res)
    e2 = exp2(res)
    sp = spillover(res)
    summary = {"exp1": e1.to_dict("records"), "exp2": e2.to_dict("records"), "spillover": sp.to_dict("records")}
    (res / "summary.json").write_text(json.dumps(summary, indent=1))

    lines = ["## Experiment 1 (mean over seeds; min-max in brackets)", "",
             "| Detector | " + " | ".join(f"AUROC k={k}" for k in exposures) + " | FPR on lockbox | "
             + " | ".join(f"TPR k={k}" for k in exposures) + " |",
             "|---" * (2 * len(exposures) + 2) + "|"]
    for d in DETECTORS + ["oracle"]:
        s = e1[e1["detector"] == d]
        cell = lambda c: f"{s[c].mean():.3f} [{s[c].min():.2f}-{s[c].max():.2f}]"  # noqa: E731
        lines.append(f"| {NAMES[d]} | " + " | ".join(cell(f"auroc_k{k}") for k in exposures) + " | "
                     + cell("fpr_lockbox") + " | " + " | ".join(cell(f"tpr_k{k}") for k in exposures) + " |")
    lines += ["", "## Experiment 2 (base model, nothing planted)", "",
              "| Detector | Within-template AUROC | AUROC originals vs rewrites | Originals flagged | All GSM8K flagged |",
              "|---|---|---|---|---|"]
    for r in e2.to_dict("records"):
        lines.append(f"| {NAMES[r['detector']]} | {fmt(r['within_template_auroc'])} "
                     f"[{r['within_template_auroc_ci'][0]:.2f}, {r['within_template_auroc_ci'][1]:.2f}] | "
                     f"{fmt(r['auroc_orig_vs_rewrites'])} | {r['flagged_originals']:.0%} | {r['flagged_all_gsm8k']:.0%} |")
    if not sp.empty:
        lines += ["", "## Spill-over to rewrites of heavily planted items (k=8,16) vs rewrites of unseen items", "",
                  "| Detector | AUROC (mean over seeds) |", "|---|---|"]
        for d in DETECTORS + ["oracle"]:
            s = sp[sp["detector"] == d]
            lines.append(f"| {NAMES[d]} | {s['auroc'].mean():.3f} [{s['auroc'].min():.2f}-{s['auroc'].max():.2f}] |")
    (res / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Figure 1: AUROC and TPR@5%FPR against exposure count
    f, ax = plt.subplots(1, 2, figsize=(10, 4))
    for d in DETECTORS + ["oracle"]:
        s = e1[e1["detector"] == d]
        style = "--" if d == "oracle" else "-"
        for a, key in zip(ax, ["auroc", "tpr"]):
            m = [s[f"{key}_k{k}"].mean() for k in exposures]
            lo = [s[f"{key}_k{k}"].min() for k in exposures]
            hi = [s[f"{key}_k{k}"].max() for k in exposures]
            a.plot(exposures, m, style, marker="o", label=NAMES[d])
            a.fill_between(exposures, lo, hi, alpha=0.15)
    ax[0].axhline(0.5, color="grey", lw=0.8)
    ax[1].axhline(ALPHA, color="grey", lw=0.8)
    for a, title in zip(ax, ["AUROC vs unseen lockbox items", "Detection rate at 5% FPR (threshold from calibration set)"]):
        a.set_xscale("log", base=2)
        a.set_xticks(exposures, [str(k) for k in exposures])
        a.set_xlabel("times the item appeared in training")
        a.set_title(title, fontsize=10)
    ax[0].set_ylabel("AUROC")
    ax[1].set_ylabel("share of planted items flagged")
    ax[1].legend(fontsize=8)
    f.tight_layout()
    f.savefig(fig / f"exp1_{args.tag}.png", dpi=160)

    # Figure 2: false alarms when rewrites are used as the unseen reference
    f, a = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(e2))
    a.bar(x, e2["flagged_originals"], color="#c0504d")
    err = np.array([[r[0] for r in e2["flagged_originals_ci"]], [r[1] for r in e2["flagged_originals_ci"]]])
    a.errorbar(x, e2["flagged_originals"], yerr=[e2["flagged_originals"] - err[0], err[1] - e2["flagged_originals"]],
               fmt="none", color="black", capsize=3)
    a.axhline(ALPHA, color="grey", ls="--", lw=1)
    a.set_xticks(x, [NAMES[d] for d in e2["detector"]])
    a.set_ylabel("unseen GSM8K originals flagged")
    a.set_title("Threshold set at 5% FPR on GSM-Symbolic rewrites (nothing planted)", fontsize=10)
    f.tight_layout()
    f.savefig(fig / f"exp2_{args.tag}.png", dpi=160)


if __name__ == "__main__":
    main()
