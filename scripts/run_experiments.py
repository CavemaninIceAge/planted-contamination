"""Score every text set with the reference, base and contaminated models.

Writes per-item detector scores (ids and numbers only, never benchmark text) to
results/<tag>/. See ANALYSIS_PLAN.md for the design.

  python scripts/run_experiments.py --pilot --lr 1e-4             # pilot on GSM8K train items only
  python scripts/run_experiments.py --lr <pilot decision> --seeds 0 1 2 3 4   # main runs
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pc import data, metrics, score, train  # noqa: E402

REVISIONS = {
    "EleutherAI/pythia-70m-deduped": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
    "EleutherAI/pythia-160m-deduped": "582159a2dfe3e712a8d47ae83dec95ae3bde8e7e",
    "EleutherAI/pythia-410m-deduped": "c4fc8d586d62df497f1f9b69d66d3ca419992d3e",
}
TRAIN_REFERENCE_SEED = 2026  # fixes which 1000 GSM8K train items serve as the in-distribution reference
TRAIN_REFERENCE_SIZE = 1000
STEP_BLOCKS = 16  # micro_batch * accum


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load(name):
    model = AutoModelForCausalLM.from_pretrained(name, revision=REVISIONS[name], dtype=torch.float32)
    return model.to(device()).eval()


def text_sets(with_questions):
    """All scored sets as a DataFrame with columns set, idx, template, instance, text."""
    tq, ta = data.load_gsm8k("test")
    rq, ra = data.load_gsm8k("train")
    sym = data.load_gsm_symbolic()
    ref_idx = np.sort(np.random.default_rng(TRAIN_REFERENCE_SEED).choice(len(rq), TRAIN_REFERENCE_SIZE, replace=False))
    tmpl = sorted(sym["original_id"].unique())
    assert all(sym.loc[sym["original_id"] == i, "original_question"].iloc[0].strip() == tq[i].strip() for i in tmpl)

    rows = []

    def add(name, idx, texts, template=-1, instance=-1):
        for j, (i, t) in enumerate(zip(idx, texts)):
            rows.append({"set": name, "idx": int(i), "text": t,
                         "template": template if np.isscalar(template) else int(template[j]),
                         "instance": instance if np.isscalar(instance) else int(instance[j])})

    test = [data.format_item(q, a) for q, a in zip(tq, ta)]
    sym_raw = [data.format_item(q, a) for q, a in zip(sym["question"], sym["answer"])]
    add("test", range(len(test)), test)
    add("train", ref_idx, [data.format_item(rq[i], ra[i]) for i in ref_idx])
    add("sym_raw", sym["original_id"], sym_raw, sym["template"].to_numpy(), sym["instance"].to_numpy())
    add("sym_norm", sym["original_id"], [data.normalize_surface(t) for t in sym_raw],
        sym["template"].to_numpy(), sym["instance"].to_numpy())
    add("orig_norm", tmpl, [data.normalize_surface(test[i]) for i in tmpl])
    if with_questions:
        add("q_test", range(len(tq)), [data.format_question(q) for q in tq])
        add("q_sym_raw", sym["original_id"], [data.format_question(q) for q in sym["question"]],
            sym["template"].to_numpy(), sym["instance"].to_numpy())
        add("q_sym_norm", sym["original_id"], [data.normalize_surface(data.format_question(q)) for q in sym["question"]],
            sym["template"].to_numpy(), sym["instance"].to_numpy())
        add("q_orig_norm", tmpl, [data.normalize_surface(data.format_question(tq[i])) for i in tmpl])
    frame = pd.DataFrame(rows)
    feats = pd.DataFrame([data.surface_features(t) for t in frame["text"]])
    return pd.concat([frame, feats], axis=1), test, (rq, ra), ref_idx


def nll_of(model, tok, texts, batch_size):
    return score.mean_nll(score.token_stats(model, tok, texts, device(), batch_size=batch_size))


def save(path, write):
    """Write through a temporary file and rename, so a file that exists is complete.

    The results disk sometimes drops out for a few minutes: retry for up to 15 minutes.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    for attempt in range(31):
        try:
            write(tmp)
            os.replace(tmp, path)
            return
        except OSError as err:
            if attempt == 30:
                raise
            print(f"write to {path} failed ({err}); retrying in 30 s", flush=True)
            time.sleep(30)


def save_json(path, obj):
    def write(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f)

    save(path, write)


def fingerprint(texts):
    return hashlib.sha256("\x00".join(texts).encode("utf-8")).hexdigest()


RUN_KEYS = ["target", "reference", "exposures", "per_group", "template_planted", "template_k", "wiki_tokens",
            "block_len", "lr"]


def free(model):
    """Drop a model and give its memory back before the next one is loaded."""
    del model
    gc.collect()
    if device().type == "mps":
        torch.mps.empty_cache()


def score_frame(model, tok, sets, ref_nll, batch_size):
    stats = score.token_stats(model, tok, sets["text"].tolist(), device(), batch_size=batch_size)
    out = sets.drop(columns="text").copy()
    for k, v in score.detector_scores(sets["text"].tolist(), stats, ref_nll=ref_nll).items():
        out[k] = v
    out["nll"] = score.mean_nll(stats)
    out["n_tokens"] = [len(s.logp) for s in stats]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="EleutherAI/pythia-160m-deduped")
    ap.add_argument("--reference", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--exposures", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--per-group", type=int, default=120)
    ap.add_argument("--template-planted", type=int, default=50)
    ap.add_argument("--template-k", type=int, default=16)
    ap.add_argument("--wiki-tokens", type=int, default=4_000_000)
    ap.add_argument("--block-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pilot", action="store_true", help="plant GSM8K train items only; no test items are touched")
    ap.add_argument("--tag", default="", help="results/<tag>/; default <model>-lr<lr>[-pilot]")
    args = ap.parse_args()

    tag = args.tag or f"{args.target.split('/')[-1]}-lr{args.lr:g}" + ("-pilot" if args.pilot else "")
    res = ROOT / "results" / tag
    res.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.target, revision=REVISIONS[args.target])
    wiki = data.load_wikitext_docs()

    if args.pilot:
        run_pilot(args, tok, wiki, res)
        return

    sets, test, _, _ = text_sets(with_questions=True)
    main_sets = sets[~sets["set"].str.startswith("q_")].reset_index(drop=True)
    texts = sets["text"].tolist()

    keys = ["set", "idx", "template", "instance"]
    assert (sets[keys].iloc[: len(main_sets)].to_numpy() == main_sets[keys].to_numpy()).all()
    # cached reference and base scores must belong to exactly these texts
    stamp = res / "texts.sha256"
    if stamp.exists():
        assert stamp.read_text().strip() == fingerprint(texts), "texts changed since the cached scores were made"
    else:
        save(stamp, lambda tmp: tmp.write_text(fingerprint(texts)))

    ref_cache = res / "ref_nll.npy"
    if not ref_cache.exists():
        model = load(args.reference)
        values = nll_of(model, tok, texts, args.batch_size)
        free(model)
        def write(tmp):
            with open(tmp, "wb") as f:
                np.save(f, values)

        save(ref_cache, write)
    ref_nll = np.load(ref_cache)
    assert len(ref_nll) == len(sets)
    main_ref = ref_nll[: len(main_sets)]

    base_path = res / "scores_base.csv.gz"
    if not base_path.exists():
        model = load(args.target)
        frame = score_frame(model, tok, sets, ref_nll, args.batch_size)
        free(model)
        save(base_path, lambda tmp: frame.to_csv(tmp, index=False, compression="gzip"))
    base = pd.read_csv(base_path)
    assert len(base) == len(sets) and (base[keys].to_numpy() == sets[keys].to_numpy()).all()
    base_nll = base["nll"].to_numpy()[: len(main_sets)]

    sym = data.load_gsm_symbolic()
    template_ids = sorted(sym["original_id"].unique())
    run = {k: vars(args)[k] for k in RUN_KEYS}
    for seed in args.seeds:
        out = res / f"scores_seed{seed}.csv.gz"
        meta = res / f"groups_seed{seed}.json"
        if out.exists() and meta.exists():
            done = {k: json.loads(meta.read_text())["args"][k] for k in RUN_KEYS}
            assert done == run, f"seed {seed} was run with {done}, not {run}"
            print(f"seed {seed}: already done", flush=True)
            continue
        groups = data.assign_groups(len(test), template_ids, args.exposures, args.per_group,
                                    args.template_planted, args.template_k, seed)
        counts = data.planted_counts(groups)
        blocks, where, bstats = data.build_blocks(tok, wiki, test, counts, args.wiki_tokens, args.block_len, seed,
                                                  multiple=STEP_BLOCKS)
        print(f"seed {seed}: {bstats}", flush=True)
        model = load(args.target)
        torch.manual_seed(seed)
        log = train.train(model, blocks, device(), peak_lr=args.lr, seed=seed)
        save_json(res / f"trainlog_seed{seed}.json", log)
        df = score_frame(model, tok, main_sets, main_ref, args.batch_size)
        free(model)
        df["oracle"] = base_nll - df["nll"]
        label = {i: g for g, idx in groups.items() for i in idx}
        df["group"] = [label.get(i, "-") if s in ("test", "orig_norm", "sym_raw", "sym_norm") else "-"
                       for s, i in zip(df["set"], df["idx"])]
        df["seed"] = seed
        steps = {int(i): sorted(p // STEP_BLOCKS for p in pos) for i, pos in where.items()}
        assert max(max(v) for v in steps.values()) < len(log), "a planted copy sits in an untrained block"
        save_json(meta, {"groups": groups, "exposure_steps": steps, "corpus": bstats,
                         "total_steps": len(log), "final_loss": log[-1]["loss"], "args": vars(args)})
        save(out, lambda tmp: df.to_csv(tmp, index=False, compression="gzip"))  # written last: marks the seed done


def run_pilot(args, tok, wiki, res):
    """Plant GSM8K *train* items (outside the 1000-item reference sample) to pick the learning rate."""
    _, _, (rq, ra), ref_idx = text_sets(with_questions=False)
    pool = np.setdiff1d(np.arange(len(rq)), ref_idx)
    items = [data.format_item(q, a) for q, a in zip(rq, ra)]
    rng = np.random.default_rng(99)
    order = rng.permutation(pool)
    groups, pos = {}, 0
    for k in args.exposures:
        groups[f"k{k}"] = order[pos : pos + args.per_group].tolist()
        pos += args.per_group
    groups["unseen"] = order[pos : pos + 600].tolist()
    counts = data.planted_counts(groups)
    blocks, _, bstats = data.build_blocks(tok, wiki, items, counts, args.wiki_tokens, args.block_len, 99,
                                          multiple=STEP_BLOCKS)
    print(f"pilot corpus: {bstats}", flush=True)
    model = load(args.target)
    save_json(res / f"trainlog_lr{args.lr:g}.json", train.train(model, blocks, device(), peak_lr=args.lr, seed=99))
    idx = [i for g in groups.values() for i in g]
    texts = [items[i] for i in idx]
    stats = score.token_stats(model, tok, texts, device(), batch_size=args.batch_size)
    ft = score.mean_nll(stats)
    free(model)
    model = load(args.target)
    base_nll = nll_of(model, tok, texts, args.batch_size)
    free(model)
    model = load(args.reference)
    ref_nll = nll_of(model, tok, texts, args.batch_size)
    free(model)
    s = score.detector_scores(texts, stats, ref_nll=ref_nll)
    s["oracle"] = base_nll - ft
    lab = np.array([g for g, ids in groups.items() for _ in ids])
    report = {}
    for d, v in s.items():
        neg = v[lab == "unseen"]
        report[d] = {k: float(metrics.auroc(v[lab == f"k{k}"], neg)) for k in args.exposures}
        print(d, {k: round(a, 4) for k, a in report[d].items()}, flush=True)
    # pilot rule, ANALYSIS_PLAN.md section 5
    if report["loss"][4] >= 0.98:
        decision = 3e-5
    elif report["loss"][16] < 0.75:
        decision = 3e-4
    else:
        decision = args.lr
    print(f"pilot rule: loss AUROC k=4 {report['loss'][4]:.4f}, k=16 {report['loss'][16]:.4f} -> lr {decision:g}",
          flush=True)
    save_json(res / f"pilot_lr{args.lr:g}.json",
              {"auroc": report, "corpus": bstats, "decision_lr": decision,
               "scores": {d: v.tolist() for d, v in s.items()}, "groups": lab.tolist()})


if __name__ == "__main__":
    main()
