"""Datasets, item formatting, group assignment, and the contaminated training corpus.

No benchmark text is stored in this repository. GSM8K (MIT) and GSM-Symbolic
(CC-BY-NC-ND-4.0) are downloaded from the Hugging Face Hub at pinned revisions.
"""

import json
import re
import unicodedata

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

GSM8K_REV = "740312add88f781978c0658806c59bc2815b9866"
SYMBOLIC_REV = "93b5b3758d9d9841ffe81d6cd2ae2b030685b078"
WIKITEXT_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"

CALC = re.compile(r"<<[^>]*>>")


def format_item(question, answer):
    """GSM8K-style item with calculator annotations removed: 'Question: ...\\nAnswer: ...'."""
    return f"Question: {question.strip()}\nAnswer: {CALC.sub('', answer).strip()}"


def format_question(question):
    return f"Question: {question.strip()}"


def normalize_surface(text):
    """Layer-B normalisation: undo GSM-Symbolic's surface changes that are not a rewrite.

    Non-dollar currency signs become '$', trailing '.0' on numbers is dropped, '÷' becomes
    '/', the blank line some rewrites put before '####' is removed, and Unicode is NFC.
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join("$" if unicodedata.category(c) == "Sc" else c for c in text)
    text = re.sub(r"(\d+)\.0\b", r"\1", text)
    text = text.replace("÷", "/")
    return re.sub(r"\n\s*\n####", "\n####", text)


def _parquet(repo, name, rev):
    return pd.read_parquet(hf_hub_download(repo, name, repo_type="dataset", revision=rev))


def load_gsm8k(split):
    df = _parquet("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", GSM8K_REV)
    return df["question"].tolist(), df["answer"].tolist()


def load_gsm_symbolic():
    """GSM-Symbolic (main): 100 GSM8K-test templates x 50 instances. Read at run time only."""
    path = hf_hub_download("apple/GSM-Symbolic", "main/test.jsonl", repo_type="dataset", revision=SYMBOLIC_REV)
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return pd.DataFrame(
        {
            "template": [r["id"] for r in rows],
            "instance": [r["instance"] for r in rows],
            "original_id": [r["original_id"] for r in rows],
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "original_question": [r["original_question"] for r in rows],
        }
    )


def load_wikitext_docs():
    """WikiText-103 (raw) training articles, split at top-level ' = Title = ' headings."""
    df = _parquet("Salesforce/wikitext", "wikitext-103-raw-v1/train-00000-of-00002.parquet", WIKITEXT_REV)
    heading = re.compile(r"^ = [^=].* = \n$")
    docs, cur = [], []
    for line in df["text"]:
        if heading.match(line) and cur:
            docs.append("".join(cur).strip())
            cur = []
        if line.strip():
            cur.append(line)
    if cur:
        docs.append("".join(cur).strip())
    return [d for d in docs if d]


def assign_groups(n_items, template_ids, exposures, per_group, n_template_planted, template_k, seed):
    """Random disjoint groups for one seed.

    Non-template items: `per_group` items at each exposure count, the rest unseen.
    Template originals (the GSM8K items GSM-Symbolic was built from): `n_template_planted`
    of them planted `template_k` times, the rest unseen. Keeping the two strata apart
    leaves the dose-response groups untouched by the spill-over design.
    """
    rng = np.random.default_rng(seed)
    templates = np.array(sorted(template_ids))
    others = np.setdiff1d(np.arange(n_items), templates)
    t_order = rng.permutation(templates)
    o_order = rng.permutation(others)
    groups = {}
    pos = 0
    for k in exposures:
        groups[f"k{k}"] = sorted(o_order[pos : pos + per_group].tolist())
        pos += per_group
    groups["unseen"] = sorted(o_order[pos:].tolist())
    groups[f"tmpl_k{template_k}"] = sorted(t_order[:n_template_planted].tolist())
    groups["tmpl_unseen"] = sorted(t_order[n_template_planted:].tolist())
    return groups


def planted_counts(groups):
    """item index -> number of planted copies."""
    counts = {}
    for name, idx in groups.items():
        m = re.fullmatch(r"(?:tmpl_)?k(\d+)", name)
        if m:
            for i in idx:
                counts[i] = int(m.group(1))
    return counts


def build_blocks(tokenizer, wiki_docs, items, counts, wiki_tokens, block_len, seed):
    """Pack WikiText plus planted items into fixed-length training blocks.

    Wiki text is streamed and cut at block edges like ordinary pretraining packing. Each
    planted copy stays whole inside one block, wrapped as [EOS] item [EOS] so it starts on a
    document boundary (matching how items are scored); copies of one item go to different
    blocks. Returns the blocks in training order, the block index of every copy, and stats.
    """
    rng = np.random.default_rng(seed)
    eos = tokenizer.eos_token_id

    wiki = []
    for i in rng.permutation(len(wiki_docs)):
        wiki.extend(tokenizer(wiki_docs[i])["input_ids"] + [eos])
        if len(wiki) >= wiki_tokens:
            break
    wiki = wiki[:wiki_tokens]

    copies = []  # (item index, token ids)
    for idx, k in sorted(counts.items()):
        ids = [eos] + tokenizer(items[idx])["input_ids"] + [eos]
        assert len(ids) <= block_len, "item longer than a training block"
        copies.extend([(idx, ids)] * k)
    planted_tokens = sum(len(ids) for _, ids in copies)
    n_blocks = (len(wiki) + planted_tokens) // block_len

    room = np.full(n_blocks, block_len)
    members = [[] for _ in range(n_blocks)]
    for c in rng.permutation(len(copies)):
        idx, ids = copies[c]
        for _ in range(10_000):
            b = int(rng.integers(n_blocks))
            if room[b] >= len(ids) and all(m[0] != idx for m in members[b]):
                members[b].append((idx, ids))
                room[b] -= len(ids)
                break
        else:
            raise RuntimeError("could not place a planted copy; lower the contamination rate")

    blocks, w = [], 0
    for b in range(n_blocks):
        fill = int(room[b])
        cuts = np.sort(rng.integers(0, fill + 1, len(members[b])))
        pieces = np.diff(np.concatenate([[0], cuts, [fill]]))
        seq = []
        for p, piece in enumerate(pieces):
            take = wiki[w : w + piece]
            if len(take) < piece:  # wiki stream exhausted: wrap around
                take = take + wiki[: piece - len(take)]
            seq.extend(take)
            w = (w + piece) % len(wiki)
            if p < len(members[b]):
                seq.extend(members[b][p][1])
        assert len(seq) == block_len
        blocks.append(seq)

    order = rng.permutation(n_blocks)
    position = np.empty(n_blocks, dtype=np.int64)
    position[order] = np.arange(n_blocks)  # where each original block lands in training order
    where = {}
    for b in range(n_blocks):
        for idx, _ in members[b]:
            where.setdefault(idx, []).append(int(position[b]))
    stats = {
        "blocks": n_blocks,
        "wiki_tokens": len(wiki),
        "planted_tokens": planted_tokens,
        "planted_copies": len(copies),
        "planted_share": planted_tokens / (n_blocks * block_len),
    }
    return np.array(blocks, dtype=np.int64)[order], where, stats
