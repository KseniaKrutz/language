# -*- coding: utf-8 -*-
"""Experiment 15: effect of Markov training-data size on BPE/Unigram.

Run from the project root:
    python src/15_training_data_scaling.py

Existing SentencePiece models are reused; tokenizers are NOT retrained.
Positive delta (BPE CE - Unigram CE) means Unigram has lower cross-entropy.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.common import CONFIG, set_seed
RESULTS = ROOT / "results"

BUDGETS = [10_000, 25_000, 50_000, 100_000, 250_000, 500_000]
ALGORITHMS = ["BPE", "Unigram"]
ALPHA = float(CONFIG.get("alpha", 0.1))
MAX_ORDER = int(CONFIG.get("max_markov_order", 4))
MAX_EVAL_SENTENCES = int(CONFIG.get("max_eval_sentences", 1000))
SEED = int((CONFIG.get("seeds") or [13])[0])
VOCAB = int((CONFIG.get("vocab_sizes") or [8000])[0])
MODELS = ROOT / "models"
SAMPLES = RESULTS / "training_samples"
MANIFEST = ROOT / "data" / "prepared_manifest.json"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def read_lines(path):
    with path.open(encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]



def manifest_entries():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        languages = data.get("languages")

        if isinstance(languages, dict):
            return [
                {"code": code, **info}
                for code, info in languages.items()
            ]

        if isinstance(languages, list):
            return languages

        for key in ("items", "corpora"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        "Unrecognized prepared_manifest.json structure"
    )



def get_code(item):
    for key in ("code", "lang", "language", "language_code"):
        if item.get(key):
            return str(item[key])
    return None


def find_test_file(code, item):
    candidates = []
    for key in ("test_path", "test_file", "test"):
        value = item.get(key)
        if isinstance(value, str):
            p = Path(value)
            candidates += [p if p.is_absolute() else ROOT / p,
                           ROOT / "data" / p]
        elif isinstance(value, dict):
            for subkey in ("path", "file"):
                if value.get(subkey):
                    p = Path(value[subkey])
                    candidates += [p if p.is_absolute() else ROOT / p,
                                   ROOT / "data" / p]
    candidates += [
        ROOT / "data" / "prepared" / f"{code}_test.txt",
        ROOT / "data" / "prepared" / code / "test.txt",
        ROOT / "data" / "processed" / f"{code}_test.txt",
        ROOT / "data" / f"{code}_test.txt",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(
        f"Test file for {code} not found. Check prepared_manifest.json "
        "and add its actual test path in find_test_file()."
    )


def get_model(code, algorithm):
    return MODELS / f"{code}_{algorithm.lower()}_v{VOCAB}_s{SEED}.model"


def make_prefix_snapshots(tokenized, budgets):
    """Count token n-grams on nested prefixes, never crossing sentence edges."""
    budgets = sorted(budgets)
    counts = {k: Counter() for k in range(MAX_ORDER + 1)}
    snapshots, actual = {}, {}
    n = 0
    next_i = 0
    for sent in tokenized:
        for pos, tok in enumerate(sent):
            if next_i == len(budgets):
                break
            for k in range(MAX_ORDER + 1):
                ctx = () if k == 0 else tuple(sent[max(0, pos-k):pos])
                counts[k][(ctx, tok)] += 1
            n += 1
            while next_i < len(budgets) and n >= budgets[next_i]:
                b = budgets[next_i]
                snapshots[b] = {k: Counter(v) for k, v in counts.items()}
                actual[b] = n
                next_i += 1
        if next_i == len(budgets):
            break
    return snapshots, actual, n


def split_counts(joint):
    ctx_totals = Counter()
    vocab = set()
    for (ctx, tok), count in joint.items():
        ctx_totals[ctx] += count
        vocab.add(tok)
    return ctx_totals, len(vocab)


def bin_name(freq):
    if freq == 0:
        return "unseen_0"
    if freq == 1:
        return "rare_1"
    if freq <= 5:
        return "low_2_5"
    if freq <= 20:
        return "medium_6_20"
    return "frequent_21_plus"


def evaluate(code, language, algorithm, budget, actual, counts, test_tokens, test_text):
    rows, bin_rows = [], []
    nbytes = sum(len(s.encode("utf-8")) for s in test_text)
    for k, joint in counts.items():
        ctx_totals, vocab_size = split_counts(joint)
        if not vocab_size:
            raise ValueError(f"Empty n-gram model: {code}/{algorithm}/k={k}")
        nll = 0.0
        n_eval = 0
        unseen = 0
        per_context_n = Counter()
        per_context_nll = Counter()
        for sent in test_tokens:
            for pos, tok in enumerate(sent):
                ctx = () if k == 0 else tuple(sent[max(0, pos-k):pos])
                cf = ctx_totals.get(ctx, 0)
                if k > 0 and cf == 0:
                    unseen += 1
                p = (joint.get((ctx, tok), 0) + ALPHA) / (cf + ALPHA * vocab_size)
                loss = -math.log2(p)
                nll += loss
                n_eval += 1
                per_context_n[ctx] += 1
                per_context_nll[ctx] += loss
        rows.append({
            "language_code": code, "language": language,
            "algorithm": algorithm, "budget_requested": budget,
            "budget_actual": actual, "alpha": ALPHA, "order": k,
            "train_context_types": len(ctx_totals), "train_joint_types": len(joint),
            "train_vocabulary_size": vocab_size, "eval_tokens": n_eval,
            "eval_utf8_bytes": nbytes,
            "cross_entropy_bits_per_byte": nll / nbytes if nbytes else float("nan"),
            "unseen_context_token_share": unseen / n_eval if k > 0 and n_eval else None,
        })
        if k == 0:
            continue
        bin_n, bin_loss = Counter(), Counter()
        for ctx, cnt in per_context_n.items():
            b = bin_name(ctx_totals.get(ctx, 0))
            bin_n[b] += cnt
            bin_loss[b] += per_context_nll[ctx]
        for b in ("unseen_0", "rare_1", "low_2_5", "medium_6_20", "frequent_21_plus"):
            bin_rows.append({
                "language_code": code, "language": language, "algorithm": algorithm,
                "budget_requested": budget, "budget_actual": actual,
                "alpha": ALPHA, "order": k, "frequency_bin": b,
                "eval_tokens": bin_n[b],
                "token_share": bin_n[b] / n_eval if n_eval else 0,
                "cross_entropy_contribution_bits_per_byte": bin_loss[b] / nbytes if nbytes else float("nan"),
            })
    return rows, bin_rows


def main():
    set_seed(SEED)
    entries = manifest_entries()
    raw_rows, bins_rows, availability = [], [], []
    for item in entries:
        code = get_code(item)
        if not code:
            continue
        language = str(item.get("name") or item.get("language_name") or code)
        train_path = SAMPLES / f"{code}_train_s{SEED}.txt"
        if not train_path.exists():
            print(f"SKIP {code}: missing saved training sample {train_path}")
            continue
        train_text = read_lines(train_path)
        test_text = read_lines(find_test_file(code, item))[:MAX_EVAL_SENTENCES]
        print(f"\\n=== {code} ({language}) ===")
        for algorithm in ALGORITHMS:
            mpath = get_model(code, algorithm)
            if not mpath.exists():
                print(f"  SKIP {algorithm}: model missing ({mpath.name})")
                continue
            sp = spm.SentencePieceProcessor(model_file=str(mpath))
            train_tok = [sp.encode(s, out_type=int) for s in train_text]
            test_tok = [sp.encode(s, out_type=int) for s in test_text]
            snapshots, actuals, total_tokens = make_prefix_snapshots(train_tok, BUDGETS)
            for budget in BUDGETS:
                ok = budget in snapshots
                availability.append({
                    "language_code": code, "language": language, "algorithm": algorithm,
                    "budget_requested": budget, "budget_available": ok,
                    "budget_actual": actuals.get(budget, total_tokens),
                    "available_training_tokens": total_tokens,
                })
                if not ok:
                    print(f"  {algorithm}: {budget:,} unavailable; only {total_tokens:,} tokens")
                    continue
                r, b = evaluate(code, language, algorithm, budget, actuals[budget],
                                snapshots[budget], test_tok, test_text)
                raw_rows.extend(r)
                bins_rows.extend(b)
                print(f"  {algorithm}: completed {budget:,} tokens")

    if not raw_rows:
        raise RuntimeError("No experiment results. Check paths, model files and manifest.")
    raw = pd.DataFrame(raw_rows)
    bins = pd.DataFrame(bins_rows)
    av = pd.DataFrame(availability)
    raw.to_csv(RESULTS / "training_data_scaling_raw.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(RESULTS / "training_data_scaling_frequency_bins.csv", index=False, encoding="utf-8-sig")
    av.to_csv(RESULTS / "training_data_scaling_availability.csv", index=False, encoding="utf-8-sig")

    summary = raw.groupby(["algorithm", "budget_requested", "order"], as_index=False).agg(
        mean_cross_entropy_bits_per_byte=("cross_entropy_bits_per_byte", "mean"),
        std_cross_entropy_bits_per_byte=("cross_entropy_bits_per_byte", "std"),
        mean_unseen_context_token_share=("unseen_context_token_share", "mean"),
        n_languages=("language_code", "nunique"),
    )
    summary.to_csv(RESULTS / "training_data_scaling_summary.csv", index=False, encoding="utf-8-sig")

    pivot = raw.pivot_table(
        index=["language_code", "language", "budget_requested", "order"],
        columns="algorithm", values="cross_entropy_bits_per_byte", aggfunc="first"
    ).reset_index()
    if {"BPE", "Unigram"}.issubset(pivot.columns):
        pivot["delta_ce_bpe_minus_unigram"] = pivot["BPE"] - pivot["Unigram"]
    pivot.to_csv(RESULTS / "training_data_scaling_paired.csv", index=False, encoding="utf-8-sig")
    plot_results(summary, pivot)
    print("\\nDone. Outputs are in results/ and results/plots/.")


def plot_results(summary, paired):
    for alg in ALGORITHMS:
        for metric, ylabel, suffix, orders in [
            ("mean_cross_entropy_bits_per_byte", "Cross-entropy (bits / UTF-8 byte)", "ce", None),
            ("mean_unseen_context_token_share", "Unseen-context token share", "unseen", (1, 2, 3, 4)),
        ]:
            sub = summary[summary.algorithm == alg]
            if orders is not None:
                sub = sub[sub.order.isin(orders)]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for k in sorted(sub.order.unique()):
                part = sub[sub.order == k].sort_values("budget_requested")
                ax.plot(part.budget_requested, part[metric], marker="o", label=f"k={k}")
            ax.set_xscale("log")
            ax.set_xlabel("Markov training budget (tokens)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{alg}: {suffix} vs training-data size")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(PLOTS / f"training_scaling_{suffix}_{alg.lower()}.png", dpi=180)
            plt.close(fig)
    if "delta_ce_bpe_minus_unigram" in paired.columns:
        d = paired.groupby(["budget_requested", "order"], as_index=False).delta_ce_bpe_minus_unigram.mean()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for k in sorted(d.order.unique()):
            part = d[d.order == k].sort_values("budget_requested")
            ax.plot(part.budget_requested, part.delta_ce_bpe_minus_unigram, marker="o", label=f"k={k}")
        ax.axhline(0, linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Markov training budget (tokens)")
        ax.set_ylabel("BPE CE - Unigram CE (bits / byte)")
        ax.set_title("Tokenizer difference vs Markov training-data size (positive favors Unigram)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS / "training_scaling_delta_bpe_minus_unigram.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
