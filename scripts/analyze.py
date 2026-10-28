"""All tables and figures in the README, computed from results/<tag>/ (see ANALYSIS_PLAN.md).

  python scripts/analyze.py --tag pythia-160m-deduped-lr0.0001

Reads only per-item scores (ids and numbers); needs no GPU and no benchmark text.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pc import metrics  # noqa: E402

DETECTORS = ["loss", "zlib", "mink", "minkpp", "ref"]
NAMES = {"loss": "Loss", "zlib": "Zlib", "mink": "Min-K%", "minkpp": "Min-K%++", "ref": "Ref (70m)",
         "oracle": "Oracle"}
COLORS = {"loss": "#2a78d6", "zlib": "#eb6834", "mink": "#1baf7a", "minkpp": "#eda100", "ref": "#e87ba4",
          "oracle": "#52514e"}
SOURCES = {"train": "unseen GSM8K train items", "sym_raw": "GSM-Symbolic rewrites",
           "sym_norm": "normalised rewrites"}
LAYERS = {"raw": ("test", "sym_raw"), "norm": ("orig_norm", "sym_norm"),
          "q_raw": ("q_test", "q_sym_raw"), "q_norm": ("q_orig_norm", "q_sym_norm")}
BLIND_FEATURES = ["n_chars", "n_tokens", "n_digits", "longest_number", "non_ascii_share", "other_currency",
                  "dot_zero"]
ALPHA = 0.05
EQUIV = 0.05
N_BOOT = 2000
N_PERM = 10000
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ---------------------------------------------------------------- loading


def load(res):
    base = pd.read_csv(res / "scores_base.csv.gz")
    seeds = {}
    for path in sorted(res.glob("scores_seed*.csv.gz")):
        df = pd.read_csv(path)
        s = int(df["seed"].iloc[0])
        seeds[s] = (df, json.loads((res / f"groups_seed{s}.json").read_text()))
    if not seeds:
        raise SystemExit(f"no scores_seed*.csv.gz in {res}")
    runs = {json.dumps(g.get("args", {}).get("lr")) for _, g in seeds.values()}
    assert len(runs) == 1, f"seeds were run with different learning rates: {runs}"
    return base, seeds


def by_template(df, orig_set, rew_set, col):
    """Template ids (= GSM8K test index of the original), original scores, rewrite score matrix."""
    rew = df[df["set"] == rew_set].sort_values(["idx", "instance"])
    ids = np.sort(rew["idx"].unique())
    width = len(rew) // len(ids)
    assert len(rew) == width * len(ids)
    assert (rew["idx"].to_numpy().reshape(len(ids), width) == ids[:, None]).all()
    orig = df[df["set"] == orig_set].set_index("idx").loc[ids, col].to_numpy(float)
    return ids, orig, rew[col].to_numpy(float).reshape(len(ids), width)


def group_scores(df, group, col, set_name="test"):
    return df.loc[(df["set"] == set_name) & (df["group"] == group), col].to_numpy(float)


# ---------------------------------------------------------------- statistics


def rank_fraction(orig, rew):
    """Per template: share of its rewrites scoring below the original (ties count half)."""
    o = orig[:, None]
    return ((rew < o).sum(1) + 0.5 * (rew == o).sum(1)) / rew.shape[1]


def own_flags(orig, rew):
    """Original flagged against a conformal threshold built from its own rewrites only."""
    return np.array([o > metrics.conformal_threshold(r, ALPHA) for o, r in zip(orig, rew)])


def pooled_flags(orig, rew):
    return orig > metrics.conformal_threshold(rew.ravel(), ALPHA)


def ci(values, level=0.95):
    lo, hi = np.percentile(values, [50 * (1 - level), 100 - 50 * (1 - level)])
    return [float(lo), float(hi)]


def template_bootstrap(fn, n_templates, seed=0, n_boot=N_BOOT, level=0.95):
    """Percentile interval of fn(rows) with whole templates resampled."""
    rng = np.random.default_rng(seed)
    return ci([fn(rng.integers(0, n_templates, n_templates)) for _ in range(n_boot)], level)


def item_bootstrap_auroc(pos, neg, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    stats = [metrics.auroc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))])
             for _ in range(N_BOOT)]
    return ci(stats, level)


def exchangeability_null(orig, rew, statistic, seed=0, n_perm=N_PERM):
    """Null distribution under within-template exchangeability of the original and its rewrites.

    Each draw picks, in every template, one of its 1 + n_rewrites texts at random to play the
    original; the rest play the rewrites. `statistic(pseudo_orig, pseudo_rew)` is recomputed.
    """
    rng = np.random.default_rng(seed)
    m = np.column_stack([orig, rew])
    n, w = m.shape
    rows = np.arange(n)
    out = np.empty(n_perm)
    for b in range(n_perm):
        pick = rng.integers(0, w, n)
        keep = np.ones_like(m, dtype=bool)
        keep[rows, pick] = False
        out[b] = statistic(m[rows, pick], m[keep].reshape(n, w - 1))
    return out


def perm_p(observed, null, two_sided=False, centre=0.0):
    if two_sided:
        return float((1 + np.sum(np.abs(null - centre) >= abs(observed - centre) - 1e-12)) / (1 + len(null)))
    return float((1 + np.sum(null >= observed - 1e-12)) / (1 + len(null)))


def adjust_for_surface(orig, rew, x_orig, x_rew):
    """Remove the within-template linear effect of covariates, fit on rewrites only.

    x_orig: (n, p), x_rew: (n, w, p). Template means are removed before the fit, so the slope
    is estimated from variation between rewrites of the same template.
    """
    yd = rew - rew.mean(1, keepdims=True)
    xd = x_rew - x_rew.mean(1, keepdims=True)
    beta, *_ = np.linalg.lstsq(xd.reshape(-1, xd.shape[-1]), yd.ravel(), rcond=None)
    return orig - x_orig @ beta, rew - x_rew @ beta, beta


def fit_logistic(x, y, lam=1.0):
    """L2-penalised logistic regression with balanced class weights (intercept unpenalised)."""
    w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))

    def loss(theta):
        z = x @ theta[1:] + theta[0]
        g = w * (expit(z) - y)
        value = np.sum(w * (np.logaddexp(0, z) - y * z)) + 0.5 * lam * np.sum(theta[1:] ** 2)
        return value, np.concatenate([[g.sum()], x.T @ g + lam * theta[1:]])

    return minimize(loss, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B").x


def blind_baseline(df, orig_set, rew_set, n_folds=5):
    """Out-of-fold scores from surface features only, 5-fold cross-validation grouped by template."""
    ids, _, _ = by_template(df, orig_set, rew_set, "n_chars")
    fold = dict(zip(ids, np.random.default_rng(0).permutation(len(ids)) % n_folds))
    rows = pd.concat([df[df["set"] == orig_set].set_index("idx").loc[ids].reset_index().assign(y=1, instance=-1),
                      df[df["set"] == rew_set].assign(y=0)], ignore_index=True)
    x = rows[BLIND_FEATURES].to_numpy(float)
    y = rows["y"].to_numpy(float)
    f = rows["idx"].map(fold).to_numpy()
    pred = np.empty(len(rows))
    for k in range(n_folds):
        tr, te = f != k, f == k
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        theta = fit_logistic((x[tr] - mu) / sd, y[tr])
        pred[te] = (x[te] - mu) / sd @ theta[1:] + theta[0]
    rows["blind"] = pred
    return rows


def mean_sd(values):
    v = np.asarray(values, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "per_seed": [float(x) for x in v]}


# ---------------------------------------------------------------- analyses


def exposures_of(seeds):
    g = next(iter(seeds.values()))[1]["groups"]
    return sorted(int(name[1:]) for name in g if name.startswith("k"))


def planted_template_group(g):
    return [name for name in g["groups"] if name.startswith("tmpl_k")][0]


def template_masks(ids, g):
    """Which templates had their original planted in this seed, and which stayed unseen."""
    planted = np.isin(ids, g["groups"][planted_template_group(g)])
    unseen = np.isin(ids, g["groups"]["tmpl_unseen"])
    assert planted.sum() + unseen.sum() == len(ids)
    return planted, unseen


def negative_controls(base, seeds):
    """Base model: GSM8K test vs train, and every planted group vs its unseen comparison before training.

    For the planted groups the AUROC is averaged over seeds; its 90% interval resamples items
    within each seed. "equivalent" means the 90% interval lies inside 0.5 +/- EQUIV.
    """
    out = {}
    test = base[base["set"] == "test"]
    train = base[base["set"] == "train"]
    g0 = next(iter(seeds.values()))[1]
    pairs = [(f"k{k}", "unseen") for k in exposures_of(seeds)] + [(planted_template_group(g0), "tmpl_unseen")]
    for d in DETECTORS:
        a = metrics.auroc(test[d].to_numpy(float), train[d].to_numpy(float))
        lo, hi = item_bootstrap_auroc(test[d].to_numpy(float), train[d].to_numpy(float), level=0.90)
        planted = {}
        for pos, neg in pairs:
            per_seed = []
            for _, g in seeds.values():
                lab = {i: name for name, idx in g["groups"].items() for i in idx}
                grp = test["idx"].map(lab).to_numpy()
                per_seed.append((test.loc[grp == pos, d].to_numpy(float), test.loc[grp == neg, d].to_numpy(float)))
            rng = np.random.default_rng(0)
            boot = [np.mean([metrics.auroc(p[rng.integers(0, len(p), len(p))], n[rng.integers(0, len(n), len(n))])
                             for p, n in per_seed]) for _ in range(N_BOOT)]
            plo, phi = ci(boot, 0.90)
            planted[pos] = {**mean_sd([metrics.auroc(p, n) for p, n in per_seed]), "ci90": [plo, phi],
                            "equivalent": bool(0.5 - EQUIV <= plo and phi <= 0.5 + EQUIV)}
        out[d] = {"auroc_test_vs_train": float(a), "ci90": [lo, hi],
                  "equivalent": bool(0.5 - EQUIV <= lo and hi <= 0.5 + EQUIV),
                  "planted_vs_unseen_before_training": planted}
    return out


def test_bed(seeds):
    """A: AUROC of each planted group vs the 619 unseen non-template items, per seed."""
    out = {}
    for d in DETECTORS + ["oracle"]:
        row = {}
        for k in exposures_of(seeds):
            row[f"k{k}"] = mean_sd([metrics.auroc(group_scores(df, f"k{k}", d), group_scores(df, "unseen", d))
                                    for df, _ in seeds.values()])
        row["template_originals"] = mean_sd([metrics.auroc(group_scores(df, planted_template_group(g), d),
                                                           group_scores(df, "unseen", d))
                                             for df, g in seeds.values()])
        out[d] = row
    return out


def threshold_source(seeds):
    """B: thresholds from three reference sets on the fine-tuned model.

    The `*_unplanted` variants (diagnostic, added after the results) build the threshold from
    the rewrites of the 50 templates whose original was NOT planted, which an auditor cannot
    do; they separate spill-over from the rest of the rewrite effect.
    """
    out = {}
    for src in list(SOURCES) + ["sym_raw_unplanted", "sym_norm_unplanted"]:
        ref_set = src.replace("_unplanted", "")
        out[src] = {}
        for d in DETECTORS + ["oracle"]:
            fpr, share, tpr = [], [], {k: [] for k in exposures_of(seeds)}
            for df, g in seeds.values():
                ref = df[df["set"] == ref_set]
                if src.endswith("_unplanted"):
                    ref = ref[ref["idx"].isin(g["groups"]["tmpl_unseen"])]
                values = ref[d].to_numpy(float)
                t = metrics.conformal_threshold(values, ALPHA)
                fpr.append(metrics.rate_above(group_scores(df, "unseen", d), t))
                for k in tpr:
                    tpr[k].append(metrics.rate_above(group_scores(df, f"k{k}", d), t))
                if src in ("sym_raw", "sym_norm") and np.any(values > t):
                    from_planted = ref["idx"].isin(g["groups"][planted_template_group(g)]).to_numpy()
                    share.append(from_planted[values > t].mean())
            entry = {"fpr_unseen": mean_sd(fpr), **{f"tpr_k{k}": mean_sd(v) for k, v in tpr.items()}}
            if share:
                entry["above_threshold_from_planted_templates"] = mean_sd(share)
            out[src][d] = entry
    return out


def clean_audit(base):
    """C: the rewrite audit on the base model, where every original is unseen."""
    out = {}
    test_all = base[base["set"] == "test"]
    for d in DETECTORS:
        out[d] = {}
        layers = {name: by_template(base, *sets, d) for name, sets in LAYERS.items()}
        ids, o, r = layers["norm"]
        _, xo_t, xr_t = by_template(base, "orig_norm", "sym_norm", "n_tokens")
        _, xo_d, xr_d = by_template(base, "orig_norm", "sym_norm", "digit_share")
        oa, ra, beta = adjust_for_surface(o, r, np.column_stack([xo_t, xo_d]), np.stack([xr_t, xr_d], -1))
        layers["adjusted"] = (ids, oa, ra)
        for name, (ids, o, r) in layers.items():
            n = len(ids)
            rf = rank_fraction(o, r)
            d_med = o - np.median(r, axis=1)
            entry = {
                "mean_rank_fraction": float(rf.mean()),
                "mean_rank_fraction_ci": template_bootstrap(lambda i: rank_fraction(o[i], r[i]).mean(), n),
                "own_rewrite_flagged": float(own_flags(o, r).mean()),
                "own_rewrite_flagged_ci": template_bootstrap(lambda i: own_flags(o[i], r[i]).mean(), n),
                "own_rewrite_flagged_expected": float(1 - np.ceil((r.shape[1] + 1) * (1 - ALPHA)) / (r.shape[1] + 1)),
                "wilcoxon_p": float(wilcoxon(d_med).pvalue) if np.any(d_med != 0) else 1.0,
                "median_diff": float(np.median(d_med)),
                "pooled_flagged": float(pooled_flags(o, r).mean()),
                "pooled_flagged_ci": template_bootstrap(lambda i: pooled_flags(o[i], r[i]).mean(), n),
            }
            if name == "raw":
                allv = test_all[d].to_numpy(float)
                rng = np.random.default_rng(1)

                def all_flagged(i):
                    t = metrics.conformal_threshold(r[i].ravel(), ALPHA)
                    return metrics.rate_above(allv[rng.integers(0, len(allv), len(allv))], t)

                entry["all_test_flagged"] = metrics.rate_above(allv, metrics.conformal_threshold(r.ravel(), ALPHA))
                entry["all_test_flagged_ci"] = template_bootstrap(all_flagged, n)
            if name == "norm":
                t_norm = metrics.conformal_threshold(r.ravel(), ALPHA)
                entry["all_test_flagged"] = metrics.rate_above(test_all[d].to_numpy(float), t_norm)
            if name == "adjusted":
                entry["beta_tokens_digit_share"] = [float(b) for b in beta]
            out[d][name] = entry
    return out


def contaminated_audit(base, seeds):
    """D and E: the rewrite audit after fine-tuning, and spill-over to rewrites.

    D is reported for raw and normalised rewrites. `*_unplanted_ref` (diagnostic, added after
    the results) uses a pooled threshold from the rewrites of unplanted templates only.
    """
    out = {}
    for d in DETECTORS:
        out[d] = {}
        for layer in ("raw", "norm"):
            m = {key: [] for key in ("recall_planted_pooled", "false_alarm_unseen_pooled", "recall_planted_own",
                                     "false_alarm_unseen_own", "rank_fraction_planted", "rank_fraction_unseen",
                                     "recall_planted_unplanted_ref", "false_alarm_unseen_unplanted_ref")}
            for df, g in seeds.values():
                ids, o, r = by_template(df, *LAYERS[layer], d)
                planted, unseen = template_masks(ids, g)
                pf, of, rf = pooled_flags(o, r), own_flags(o, r), rank_fraction(o, r)
                pu = o > metrics.conformal_threshold(r[unseen].ravel(), ALPHA)
                for key, values, mask in (("recall_planted_pooled", pf, planted), ("false_alarm_unseen_pooled", pf, unseen),
                                          ("recall_planted_own", of, planted), ("false_alarm_unseen_own", of, unseen),
                                          ("rank_fraction_planted", rf, planted), ("rank_fraction_unseen", rf, unseen),
                                          ("recall_planted_unplanted_ref", pu, planted),
                                          ("false_alarm_unseen_unplanted_ref", pu, unseen)):
                    m[key].append(values[mask].mean())
            out[d][layer] = {key: mean_sd(v) for key, v in m.items()}

        did, auc_delta, deltas = [], [], []
        ids_base, _, r_base = by_template(base, "test", "sym_raw", d)
        for df, g in seeds.values():
            ids, _, r = by_template(df, "test", "sym_raw", d)
            assert (ids == ids_base).all()
            planted, unseen = template_masks(ids, g)
            delta = (r - r_base).mean(1)  # per template: mean score gain of its rewrites from fine-tuning
            deltas.append((delta, planted))
            did.append(delta[planted].mean() - delta[unseen].mean())
            auc_delta.append(metrics.auroc(delta[planted], delta[unseen]))
        rng = np.random.default_rng(2)
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            vals = []
            for delta, planted in deltas:
                p = rng.permutation(planted)
                vals.append(delta[p].mean() - delta[~p].mean())
            null[b] = np.mean(vals)
        out[d].update(spillover_did=mean_sd(did), spillover_auroc=mean_sd(auc_delta),
                      spillover_p=perm_p(float(np.mean(did)), null))
    return out


def hypotheses(base, contaminated):
    out = {}
    h1, h2 = [], []
    for d in DETECTORS:
        _, o, r = by_template(base, "test", "sym_raw", d)
        obs = pooled_flags(o, r).mean()
        h1.append((obs, perm_p(obs, exchangeability_null(o, r, lambda a, b: pooled_flags(a, b).mean(), seed=3))))
        _, o, r = by_template(base, "orig_norm", "sym_norm", d)
        obs = rank_fraction(o, r).mean()
        null = exchangeability_null(o, r, lambda a, b: rank_fraction(a, b).mean(), seed=4)
        h2.append((obs, perm_p(obs, null, two_sided=True, centre=0.5)))
    h3 = [(contaminated[d]["spillover_did"]["mean"], contaminated[d]["spillover_p"]) for d in DETECTORS]
    for name, family in (("H1", h1), ("H2", h2), ("H3", h3)):
        adj = metrics.holm([p for _, p in family])
        out[name] = {d: {"statistic": float(s), "p": float(p), "p_holm": float(a)}
                     for d, (s, p), a in zip(DETECTORS, family, adj)}
    return out


def blind(base):
    """Surface features only, originals vs rewrites: pooled AUROC and the within-template rank used in C."""
    out = {}
    for name in ("raw", "norm"):
        rows = blind_baseline(base, *LAYERS[name])
        ids, o, r = by_template(rows.assign(set=np.where(rows["y"] == 1, "o", "r")), "o", "r", "blind")
        entry = {"blind_auroc": float(metrics.auroc(o, r.ravel())),
                 "blind_auroc_ci": template_bootstrap(lambda i: metrics.auroc(o[i], r[i].ravel()), len(ids)),
                 "blind_mean_rank_fraction": float(rank_fraction(o, r).mean()),
                 "blind_mean_rank_fraction_ci": template_bootstrap(lambda i: rank_fraction(o[i], r[i]).mean(), len(ids))}
        for d in DETECTORS:
            _, od, rd = by_template(base, *LAYERS[name], d)
            entry[f"{d}_auroc"] = float(metrics.auroc(od, rd.ravel()))
            entry[f"{d}_auroc_ci"] = template_bootstrap(lambda i: metrics.auroc(od[i], rd[i].ravel()), len(ids))
            entry[f"{d}_mean_rank_fraction"] = float(rank_fraction(od, rd).mean())
        out[name] = entry
    return out


def k1_timing(seeds):
    """Exploratory: does a single exposure early or late in training matter? (train-reference threshold)"""
    out = {}
    if 1 not in exposures_of(seeds):
        return out
    for d in DETECTORS + ["oracle"]:
        early, late = [], []
        for df, g in seeds.values():
            t = metrics.conformal_threshold(df.loc[df["set"] == "train", d].to_numpy(float), ALPHA)
            half = g["total_steps"] / 2
            steps = {int(i): v[0] for i, v in g["exposure_steps"].items() if len(v) == 1}
            k1 = df[(df["set"] == "test") & (df["group"] == "k1")]
            when = k1["idx"].map(steps).to_numpy()
            early.append(metrics.rate_above(k1.loc[when < half, d], t))
            late.append(metrics.rate_above(k1.loc[when >= half, d], t))
        out[d] = {"early_half": mean_sd(early), "late_half": mean_sd(late)}
    return out


# ---------------------------------------------------------------- report


def f3(x):
    return f"{x:.3f}"


def ms(entry, pct=False):
    if pct:
        return f"{100 * entry['mean']:.1f}% ± {100 * entry['sd']:.1f}"
    return f"{entry['mean']:.3f} ± {entry['sd']:.3f}"


def pct(x):
    return f"{100 * x:.1f}%"


def ci_pct(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def ci3(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


LAYER_NAMES = {"raw": "raw", "norm": "normalised", "adjusted": "normalised + adjusted",
               "q_raw": "question only, raw", "q_norm": "question only, normalised"}
SOURCE_ROWS = {"train": "unseen GSM8K train items", "sym_raw": "GSM-Symbolic rewrites",
               "sym_norm": "normalised rewrites",
               "sym_raw_unplanted": "rewrites of unplanted templates only (diagnostic)",
               "sym_norm_unplanted": "normalised rewrites of unplanted templates only (diagnostic)"}


def markdown(s, ks, n_seeds):
    L = []
    tk = next(k for k in s["negative_controls"]["loss"]["planted_vs_unseen_before_training"] if k.startswith("tmpl"))
    L += [f"Results over {n_seeds} seeds: mean ± standard deviation across seeds, each metric computed within a seed.",
          "", "### Negative controls (base model, nothing planted)", "",
          "| Detector | AUROC GSM8K test vs train [90% CI] | within 0.5 ± 0.05 |", "|---|---|---|"]
    for d in DETECTORS:
        e = s["negative_controls"][d]
        L.append(f"| {NAMES[d]} | {f3(e['auroc_test_vs_train'])} {ci3(e['ci90'])} | {'yes' if e['equivalent'] else 'no'} |")
    L += ["", "AUROC of each group-to-be-planted vs its unseen comparison, before any training "
              "[90% CI; * = interval not inside 0.5 ± 0.05]:", "",
          "| Detector | " + " | ".join(f"k={k}" for k in ks) + " | template originals |", "|---" * (len(ks) + 2) + "|"]
    for d in DETECTORS:
        e = s["negative_controls"][d]["planted_vs_unseen_before_training"]
        cells = [f"{x['mean']:.3f} [{x['ci90'][0]:.2f}, {x['ci90'][1]:.2f}]{'' if x['equivalent'] else ' *'}"
                 for x in [e[f"k{k}"] for k in ks] + [e[tk]]]
        L.append(f"| {NAMES[d]} | " + " | ".join(cells) + " |")
    L += ["", "### A. Test bed: AUROC of planted items vs the 619 unseen test items, by number of exposures", "",
          "| Detector | " + " | ".join(f"k={k}" for k in ks) + " | template originals, k=16 |",
          "|---" * (len(ks) + 2) + "|"]
    for d in DETECTORS + ["oracle"]:
        e = s["test_bed"][d]
        L.append(f"| {NAMES[d]} | " + " | ".join(ms(e[f'k{k}']) for k in ks) + f" | {ms(e['template_originals'])} |")
    L += ["", "### B. Where the threshold comes from (fine-tuned model, 5% target false-positive rate)", "",
          "| Threshold from | Detector | Unseen test items flagged | " + " | ".join(
              f"detected, k={k}" for k in ks) + " |", "|---" * (len(ks) + 3) + "|"]
    for src, label in SOURCE_ROWS.items():
        for d in DETECTORS + ["oracle"]:
            e = s["threshold_source"][src][d]
            L.append(f"| {label} | {NAMES[d]} | {ms(e['fpr_unseen'], True)} | "
                     + " | ".join(ms(e[f'tpr_k{k}'], True) for k in ks) + " |")
    L += ["", "Share of the rewrites above the threshold that belong to the 50 planted templates:", "",
          "| Detector | raw rewrites | normalised rewrites |", "|---|---|---|"]
    for d in DETECTORS:
        cells = [ms(s["threshold_source"][src][d]["above_threshold_from_planted_templates"], True)
                 for src in ("sym_raw", "sym_norm")]
        L.append(f"| {NAMES[d]} | " + " | ".join(cells) + " |")
    L += ["", "### C. Rewrite audit on the clean base model (all 100 originals unseen)", "",
          "| Detector | Rewrites | Mean rank of original among its rewrites [95% CI] | Flagged vs own rewrites [95% CI] | "
          "Flagged by pooled rewrite threshold [95% CI] | All 1,319 test items flagged | Wilcoxon p |",
          "|---|---|---|---|---|---|---|"]
    for d in DETECTORS:
        for layer, label in LAYER_NAMES.items():
            e = s["clean_audit"][d][layer]
            allf = pct(e["all_test_flagged"]) if "all_test_flagged" in e else "-"
            L.append(f"| {NAMES[d]} | {label} | {f3(e['mean_rank_fraction'])} {ci3(e['mean_rank_fraction_ci'])} | "
                     f"{pct(e['own_rewrite_flagged'])} {ci_pct(e['own_rewrite_flagged_ci'])} | "
                     f"{pct(e['pooled_flagged'])} {ci_pct(e['pooled_flagged_ci'])} | {allf} | {e['wilcoxon_p']:.2g} |")
    exp_own = s["clean_audit"][DETECTORS[0]]["raw"]["own_rewrite_flagged_expected"]
    L += ["", f"Under exchangeability the mean rank is 0.5, {pct(exp_own)} of originals are flagged against their own "
              "rewrites, and at most 5% by the pooled threshold.", "",
          "### D. Rewrite audit on the contaminated model (50 originals planted 16 times, 50 unseen)", "",
          "| Detector | Rewrites | Pooled threshold: planted caught | Pooled threshold: unseen flagged | "
          "Own rewrites: planted caught | Own rewrites: unseen flagged | Mean rank, planted | Mean rank, unseen |",
          "|---|---|---|---|---|---|---|---|"]
    for d in DETECTORS:
        for layer in ("raw", "norm"):
            e = s["contaminated_audit"][d][layer]
            L.append(f"| {NAMES[d]} | {LAYER_NAMES[layer]} | {ms(e['recall_planted_pooled'], True)} | "
                     f"{ms(e['false_alarm_unseen_pooled'], True)} | {ms(e['recall_planted_own'], True)} | "
                     f"{ms(e['false_alarm_unseen_own'], True)} | {ms(e['rank_fraction_planted'])} | "
                     f"{ms(e['rank_fraction_unseen'])} |")
    L += ["", "Diagnostic: pooled threshold from the rewrites of the 50 unplanted templates only (an auditor cannot do this):",
          "", "| Detector | Rewrites | planted caught | unseen flagged |", "|---|---|---|---|"]
    for d in DETECTORS:
        for layer in ("raw", "norm"):
            e = s["contaminated_audit"][d][layer]
            L.append(f"| {NAMES[d]} | {LAYER_NAMES[layer]} | {ms(e['recall_planted_unplanted_ref'], True)} | "
                     f"{ms(e['false_alarm_unseen_unplanted_ref'], True)} |")
    L += ["", "### E. Spill-over: score gain of rewrites, planted templates minus unplanted templates", "",
          "| Detector | Difference in differences | AUROC of per-template gain, planted vs unplanted | permutation p |",
          "|---|---|---|---|"]
    for d in DETECTORS:
        e = s["contaminated_audit"][d]
        L.append(f"| {NAMES[d]} | {ms(e['spillover_did'])} | {ms(e['spillover_auroc'])} | {e['spillover_p']:.2g} |")
    L += ["", "Loss and Ref give the same gain by construction (the reference model's term cancels), and both equal "
              "the Oracle score of the rewrites.", "",
          "### Pre-registered hypotheses (Holm-corrected over the five detectors)", "",
          "| Hypothesis | Detector | Statistic | p | Holm p |", "|---|---|---|---|---|"]
    labels = {"H1": "H1 clean originals flagged by raw-rewrite threshold, above exchangeable rate",
              "H2": "H2 mean rank ≠ 0.5 (normalised)", "H3": "H3 spill-over > 0"}
    for h, label in labels.items():
        for d in DETECTORS:
            e = s["hypotheses"][h][d]
            L.append(f"| {label} | {NAMES[d]} | {e['statistic']:.3g} | {e['p']:.2g} | {e['p_holm']:.2g} |")
    L += ["", "### Blind baseline: surface features only, originals vs rewrites (base model)", "",
          "| Rewrites | Classifier: pooled AUROC [95% CI] | Classifier: mean rank among own rewrites [95% CI] | "
          + " | ".join(f"{NAMES[d]}: pooled AUROC / mean rank" for d in DETECTORS) + " |",
          "|---" * (len(DETECTORS) + 3) + "|"]
    for layer in ("raw", "norm"):
        e = s["blind_baseline"][layer]
        L.append(f"| {LAYER_NAMES[layer]} | {f3(e['blind_auroc'])} {ci3(e['blind_auroc_ci'])} | "
                 f"{f3(e['blind_mean_rank_fraction'])} {ci3(e['blind_mean_rank_fraction_ci'])} | "
                 + " | ".join(f"{f3(e[f'{d}_auroc'])} / {f3(e[f'{d}_mean_rank_fraction'])}" for d in DETECTORS) + " |")
    if s["k1_timing"]:
        L += ["", "### Exploratory: single exposure early vs late in training (train-reference threshold)", "",
              "| Detector | detected, first half | detected, second half |", "|---|---|---|"]
        for d in DETECTORS + ["oracle"]:
            e = s["k1_timing"][d]
            L.append(f"| {NAMES[d]} | {ms(e['early_half'], True)} | {ms(e['late_half'], True)} |")
    return "\n".join(L) + "\n"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bars(ax, groups, values, errors=None):
    """Grouped bars: one group per x position, one bar per detector, detector colours fixed."""
    n = len(DETECTORS)
    width = 0.8 / n
    for j, d in enumerate(DETECTORS):
        x = np.arange(len(groups)) + (j - (n - 1) / 2) * width
        ax.bar(x, values[d], width, color=COLORS[d], edgecolor=SURFACE, linewidth=1.5, label=NAMES[d],
               yerr=None if errors is None else errors[d], error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    ax.set_xticks(np.arange(len(groups)), groups)


def figures(s, ks, fig_dir, tag):
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK})

    # Figure 1: threshold source
    f, ax = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    f.patch.set_facecolor(SURFACE)
    srcs = ["train", "sym_raw", "sym_norm", "sym_raw_unplanted"]
    src_labels = ["unseen GSM8K\ntrain items", "GSM-Symbolic\nrewrites", "normalised\nrewrites",
                  "rewrites of unplanted\ntemplates (diagnostic)"]
    vals = {d: [100 * s["threshold_source"][src][d]["fpr_unseen"]["mean"] for src in srcs] for d in DETECTORS}
    errs = {d: [100 * s["threshold_source"][src][d]["fpr_unseen"]["sd"] for src in srcs] for d in DETECTORS}
    bars(ax[0], src_labels, vals, errs)
    ax[0].axhline(100 * ALPHA, color=INK2, ls="--", lw=1)
    ax[0].set_ylabel("unseen test items flagged (%)\nmean ± SD over seeds")
    ax[0].set_title("(a) False positives, by where the threshold comes from", fontsize=9, loc="left")
    ax[0].tick_params(axis="x", labelsize=7)
    for a, src, title in ((ax[1], "train", "(b) Detected, threshold from unseen train items"),
                          (ax[2], "sym_raw", "(c) Detected, threshold from all rewrites")):
        for d in DETECTORS + ["oracle"]:
            e = s["threshold_source"][src][d]
            y = [100 * e[f"tpr_k{k}"]["mean"] for k in ks]
            a.plot(ks, y, color=COLORS[d], lw=2, ls="--" if d == "oracle" else "-", marker="o", ms=4,
                   label=NAMES[d] + (" (NLL before - after)" if d == "oracle" else ""))
        a.set_xscale("log", base=2)
        a.set_xticks(ks, [str(k) for k in ks])
        a.set_ylim(0, 100)
        a.set_xlabel("times the item appeared in training")
        a.set_ylabel("planted items flagged (%), mean over seeds")
        a.set_title(title, fontsize=9, loc="left")
    for a in ax:
        style(a)
    ax[2].legend(fontsize=7, frameon=False, loc="upper left")
    f.tight_layout()
    f.savefig(fig_dir / f"threshold_source_{tag}.png", dpi=170, facecolor=SURFACE)
    plt.close(f)

    # Figure 2: clean-model audit
    f, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    f.patch.set_facecolor(SURFACE)
    layers = ["raw", "norm", "adjusted"]
    labels = ["raw rewrites", "normalised", "normalised +\nadjusted"]
    ca = s["clean_audit"]

    def bar_ci(key):
        vals = {d: [100 * ca[d][layer][key] for layer in layers] for d in DETECTORS}
        errs = {d: np.array([[100 * (ca[d][layer][key] - ca[d][layer][key + "_ci"][0]) for layer in layers],
                             [100 * (ca[d][layer][key + "_ci"][1] - ca[d][layer][key]) for layer in layers]])
                for d in DETECTORS}
        return vals, errs

    vals, errs = bar_ci("own_rewrite_flagged")
    bars(ax[0], labels, vals, errs)
    ax[0].axhline(100 * ca["loss"]["raw"]["own_rewrite_flagged_expected"], color=INK2, ls="--", lw=1)
    ax[0].set_ylabel("unseen originals flagged (%)")
    ax[0].set_title("(a) Clean model: originals flagged against their own 50 rewrites\n"
                    "dashed: rate if originals and rewrites were exchangeable; bars: 95% template-bootstrap CI",
                    fontsize=8.5, loc="left")
    vals = {d: [ca[d][layer]["mean_rank_fraction"] for layer in layers] for d in DETECTORS}
    errs = {d: np.array([[ca[d][layer]["mean_rank_fraction"] - ca[d][layer]["mean_rank_fraction_ci"][0] for layer in layers],
                         [ca[d][layer]["mean_rank_fraction_ci"][1] - ca[d][layer]["mean_rank_fraction"] for layer in layers]])
            for d in DETECTORS}
    bars(ax[1], labels, vals, errs)
    ax[1].axhline(0.5, color=INK2, ls="--", lw=1)
    ax[1].set_ylim(0, 1.08)
    ax[1].set_ylabel("mean rank of original among its rewrites")
    ax[1].set_title("(b) Clean model: where originals rank (0.5 = no difference)\nbars: 95% template-bootstrap CI",
                    fontsize=8.5, loc="left")
    for a in ax:
        style(a)
    ax[1].legend(fontsize=7, frameon=False, loc="upper center", ncol=5)
    f.tight_layout()
    f.savefig(fig_dir / f"clean_audit_{tag}.png", dpi=170, facecolor=SURFACE)
    plt.close(f)

    # Figure 3: dose-response (test bed check)
    f, a = plt.subplots(figsize=(5.4, 3.8))
    f.patch.set_facecolor(SURFACE)
    for d in DETECTORS + ["oracle"]:
        e = s["test_bed"][d]
        m = np.array([e[f"k{k}"]["mean"] for k in ks])
        sd = np.array([e[f"k{k}"]["sd"] for k in ks])
        a.plot(ks, m, color=COLORS[d], lw=2, ls="--" if d == "oracle" else "-", marker="o", ms=4, label=NAMES[d])
        a.fill_between(ks, m - sd, m + sd, color=COLORS[d], alpha=0.12, linewidth=0)
    a.axhline(0.5, color=INK2, lw=0.8)
    a.set_xscale("log", base=2)
    a.set_xticks(ks, [str(k) for k in ks])
    a.set_ylim(0.4, 1.0)
    a.set_xlabel("times the item appeared in training")
    a.set_ylabel("AUROC, planted vs unseen test items\n(band: ± 1 SD over seeds)")
    style(a)
    a.legend(fontsize=7, frameon=False, loc="lower right")
    f.tight_layout()
    f.savefig(fig_dir / f"dose_response_{tag}.png", dpi=170, facecolor=SURFACE)
    plt.close(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pythia-160m-deduped-lr0.0001")
    ap.add_argument("--results", default=None, help="results directory (default results/<tag>)")
    ap.add_argument("--figures", default=None, help="figure directory (default figures/)")
    args = ap.parse_args()
    res = Path(args.results) if args.results else ROOT / "results" / args.tag
    base, seeds = load(res)
    ks = exposures_of(seeds)
    contaminated = contaminated_audit(base, seeds)
    summary = {
        "seeds": sorted(seeds),
        "negative_controls": negative_controls(base, seeds),
        "test_bed": test_bed(seeds),
        "threshold_source": threshold_source(seeds),
        "clean_audit": clean_audit(base),
        "contaminated_audit": contaminated,
        "hypotheses": hypotheses(base, contaminated),
        "blind_baseline": blind(base),
        "k1_timing": k1_timing(seeds),
    }
    (res / "summary.json").write_text(json.dumps(summary, indent=1))
    md = markdown(summary, ks, len(seeds))
    (res / "summary.md").write_text(md)
    figures(summary, ks, Path(args.figures) if args.figures else ROOT / "figures", args.tag)
    print(md)


if __name__ == "__main__":
    main()
