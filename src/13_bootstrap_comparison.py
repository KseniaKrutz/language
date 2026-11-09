"""
13_bootstrap_comparison.py
Paired sentence-level bootstrap comparison of SentencePiece BPE and Unigram.

Uses the already trained models, saved training samples, prepared test corpora,
and the same empirical Markov-counting procedure as 03_train_and_analyze.py.

Primary comparison:
    delta = cross_entropy_BPE_bits_per_UTF8_byte
            - cross_entropy_Unigram_bits_per_UTF8_byte

Positive delta => Unigram has lower cross-entropy.
Negative delta => BPE has lower cross-entropy.

No models are trained or overwritten.
"""

import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2

from common import ROOT, CONFIG


# -----------------------------
# Settings
# -----------------------------
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
TRAINING_SAMPLES = RESULTS / "training_samples"

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261110
ALPHAS = [0.01, 0.1, 0.2, 0.5, 1.0]
MAX_TRAIN_SENTENCES = 20000
PLOT_ALPHA = 0.1


def read_sentences(path, limit=None):
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()
            if sentence:
                sentences.append(sentence)
            if limit is not None and len(sentences) >= limit:
                break
    return sentences


def get_emittable_vocab_size(sp):
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(sp.serialized_model_proto())
    size = sum(1 for piece in proto.pieces if piece.type in (1, 2))
    if size < 2:
        raise ValueError(f"Invalid emittable vocabulary size: {size}")
    return size


def fit_ngram_counts(sequences, max_order, max_tokens, vocab_size):
    counts = [Counter() for _ in range(max_order + 1)]
    context_counts = [Counter() for _ in range(max_order + 1)]
    seen = 0

    for sequence in sequences:
        for i, token in enumerate(sequence):
            if seen >= max_tokens:
                break

            counts[0][(token,)] += 1

            for k in range(1, max_order + 1):
                if i >= k:
                    context = tuple(sequence[i-k:i])
                    counts[k][(context, token)] += 1
                    context_counts[k][context] += 1

            seen += 1

        if seen >= max_tokens:
            break

    return counts, context_counts, max(vocab_size, 2)


def sentence_nll_by_order(sequence, counts, context_counts, vocab_size,
                          max_order, alpha):
    """Return total NLL in bits for each Markov order, for one sentence."""
    unigram_total = sum(counts[0].values())
    result = np.zeros(max_order + 1, dtype=float)

    for k in range(max_order + 1):
        total_nll = 0.0

        for i, token in enumerate(sequence):
            if k == 0:
                observed = counts[0].get((token,), 0)
                denominator = unigram_total + alpha * vocab_size
            else:
                context = tuple(sequence[max(0, i-k):i])
                effective_order = len(context)

                if effective_order == 0:
                    observed = counts[0].get((token,), 0)
                    denominator = unigram_total + alpha * vocab_size
                else:
                    observed = counts[effective_order].get(
                        (context, token), 0
                    )
                    context_total = context_counts[effective_order].get(
                        context, 0
                    )
                    denominator = context_total + alpha * vocab_size

            probability = (observed + alpha) / denominator
            if not 0 < probability <= 1:
                raise ValueError(f"Invalid probability: {probability}")
            total_nll -= math.log2(probability)

        result[k] = total_nll

    return result


def evaluate_model(sp, train_sentences, test_sentences,
                   max_order, max_train_tokens, alphas):
    train_sequences = [
        sp.encode(sentence, out_type=int)
        for sentence in train_sentences
    ]
    test_sequences = [
        sp.encode(sentence, out_type=int)
        for sentence in test_sentences
    ]

    vocab_size = get_emittable_vocab_size(sp)
    counts, context_counts, vocab_size = fit_ngram_counts(
        train_sequences,
        max_order,
        max_train_tokens,
        vocab_size
    )

    # nll[alpha][order] is a vector: one NLL value per test sentence.
    nll = {
        alpha: np.zeros((max_order + 1, len(test_sequences)), dtype=float)
        for alpha in alphas
    }

    for sentence_index, sequence in enumerate(test_sequences):
        for alpha in alphas:
            nll[alpha][:, sentence_index] = sentence_nll_by_order(
                sequence,
                counts,
                context_counts,
                vocab_size,
                max_order,
                alpha
            )

    return nll


def bootstrap_language(code, meta, seed, vocab_size_requested,
                       test_sentences, train_sentences, max_order,
                       max_train_tokens, rng):
    model_results = {}

    for algorithm in ("bpe", "unigram"):
        model_path = (
            MODELS
            / f"{code}_{algorithm}_v{vocab_size_requested}_s{seed}.model"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        model_results[algorithm] = evaluate_model(
            sp,
            train_sentences,
            test_sentences,
            max_order,
            max_train_tokens,
            ALPHAS
        )

    byte_lengths = np.array(
        [len(sentence.encode("utf-8")) for sentence in test_sentences],
        dtype=float
    )
    n_sentences = len(test_sentences)

    if n_sentences < 2:
        raise ValueError(f"Too few test sentences for {code}: {n_sentences}")

    # The same bootstrap indices are used for both tokenizers and all metrics.
    indices = rng.integers(
        0, n_sentences, size=(N_BOOTSTRAP, n_sentences)
    )
    bootstrap_byte_totals = byte_lengths[indices].sum(axis=1)

    rows = []

    for alpha in ALPHAS:
        bpe_matrix = model_results["bpe"][alpha]
        unigram_matrix = model_results["unigram"][alpha]

        for order in range(max_order + 1):
            bpe_sentence_nll = bpe_matrix[order]
            unigram_sentence_nll = unigram_matrix[order]

            observed_bpe = bpe_sentence_nll.sum() / byte_lengths.sum()
            observed_unigram = unigram_sentence_nll.sum() / byte_lengths.sum()
            observed_delta = observed_bpe - observed_unigram

            # Paired resampling: identical sentence indices for BPE and Unigram.
            bpe_boot = bpe_sentence_nll[indices].sum(axis=1) / bootstrap_byte_totals
            unigram_boot = (
                unigram_sentence_nll[indices].sum(axis=1)
                / bootstrap_byte_totals
            )
            delta_boot = bpe_boot - unigram_boot

            ci_low, ci_high = np.percentile(delta_boot, [2.5, 97.5])

            if ci_low > 0:
                interpretation = "unigram_lower_cross_entropy"
            elif ci_high < 0:
                interpretation = "bpe_lower_cross_entropy"
            else:
                interpretation = "interval_includes_zero"

            rows.append({
                "language": meta["name"],
                "code": code,
                "typology_group": meta["group"],
                "seed": seed,
                "vocab_size_requested": vocab_size_requested,
                "alpha": alpha,
                "markov_order": order,
                "n_test_sentences": n_sentences,
                "observed_bpe_bits_per_utf8_byte": observed_bpe,
                "observed_unigram_bits_per_utf8_byte": observed_unigram,
                "delta_bpe_minus_unigram": observed_delta,
                "bootstrap_ci_2_5": ci_low,
                "bootstrap_ci_97_5": ci_high,
                "bootstrap_probability_delta_positive": float(
                    np.mean(delta_boot > 0)
                ),
                "bootstrap_probability_delta_negative": float(
                    np.mean(delta_boot < 0)
                ),
                "bootstrap_repetitions": N_BOOTSTRAP,
                "ci_interpretation": interpretation
            })

    return rows


def make_plots(summary):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    selected = summary[np.isclose(summary["alpha"], PLOT_ALPHA)].copy()
    selected = selected[selected["markov_order"] == 4].copy()

    if not selected.empty:
        selected = selected.sort_values("delta_bpe_minus_unigram")
        y = np.arange(len(selected))
        x = selected["delta_bpe_minus_unigram"].to_numpy()
        low = selected["bootstrap_ci_2_5"].to_numpy()
        high = selected["bootstrap_ci_97_5"].to_numpy()

        plt.figure(figsize=(10, max(6, len(selected) * 0.35)))
        plt.errorbar(
            x, y,
            xerr=[
                np.maximum(0, x - low),
                np.maximum(0, high - x)
            ],
            fmt="o",
            capsize=3
        )
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.yticks(y, selected["language"])
        plt.xlabel("BPE − Unigram cross-entropy (bits / UTF-8 byte)")
        plt.title(f"Paired bootstrap comparison, Markov order 4, alpha={PLOT_ALPHA}")
        plt.tight_layout()
        plt.savefig(RESULTS / "bootstrap_forest_k4_alpha_0.1.png", dpi=180)
        plt.close()

    heat = selected.pivot(
        index="language",
        columns="markov_order",
        values="delta_bpe_minus_unigram"
    ) if not selected.empty else None

    # A second plot: confidence interval width by language at the primary alpha.
    width = summary[
        np.isclose(summary["alpha"], PLOT_ALPHA)
    ].copy()
    if not width.empty:
        width["ci_width"] = (
            width["bootstrap_ci_97_5"] - width["bootstrap_ci_2_5"]
        )
        width = width.groupby("language", as_index=False)["ci_width"].mean()
        width = width.sort_values("ci_width")

        plt.figure(figsize=(9, max(5, len(width) * 0.35)))
        plt.barh(width["language"], width["ci_width"])
        plt.xlabel("Mean 95% bootstrap interval width")
        plt.title(f"Bootstrap uncertainty by language, alpha={PLOT_ALPHA}")
        plt.tight_layout()
        plt.savefig(RESULTS / "bootstrap_ci_width_alpha_0.1.png", dpi=180)
        plt.close()


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / "data" / "prepared_manifest.json"
    metrics_path = RESULTS / "metrics_raw.csv"

    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    if not metrics_path.exists():
        raise SystemExit(f"Missing core results: {metrics_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)

    required_columns = {
        "code", "seed", "vocab_size_requested", "algorithm"
    }
    missing = required_columns - set(metrics.columns)
    if missing:
        raise SystemExit(f"metrics_raw.csv is missing columns: {sorted(missing)}")

    # Only process language/seed/vocabulary combinations actually present
    # in the completed core experiment (avoids planned but unrun languages).
    combinations = (
        metrics[["code", "seed", "vocab_size_requested"]]
        .drop_duplicates()
        .sort_values(["code", "seed", "vocab_size_requested"])
    )

    exp = CONFIG["experiment"]
    max_order = int(exp["max_markov_order"])
    max_train_tokens = int(exp["max_ngram_train_tokens"])
    max_eval_sentences = int(exp["max_eval_sentences"])

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    all_rows = []

    for row in combinations.itertuples(index=False):
        code = str(row.code)
        seed = int(row.seed)
        vocab_size_requested = int(row.vocab_size_requested)

        if code not in manifest["languages"]:
            print(f"Skipping {code}: not found in prepared manifest.")
            continue

        meta = manifest["languages"][code]
        test_path = ROOT / meta["test_file"]
        sample_path = (
            TRAINING_SAMPLES / f"{code}_train_s{seed}.txt"
        )

        if not test_path.exists():
            print(f"Skipping {code}: missing test file {test_path}")
            continue
        if not sample_path.exists():
            print(f"Skipping {code}: missing training sample {sample_path}")
            continue

        test_sentences = read_sentences(test_path, max_eval_sentences)
        train_sentences = read_sentences(
            sample_path, MAX_TRAIN_SENTENCES
        )

        if not test_sentences or not train_sentences:
            print(f"Skipping {code}: empty train/test data.")
            continue

        print(
            f"\n=== {meta['name']} ({code}), seed={seed}, "
            f"vocab={vocab_size_requested} ==="
        )
        rows = bootstrap_language(
            code,
            meta,
            seed,
            vocab_size_requested,
            test_sentences,
            train_sentences,
            max_order,
            max_train_tokens,
            rng
        )
        all_rows.extend(rows)
        print(f"Completed {len(rows)} comparisons.")

    if not all_rows:
        raise SystemExit(
            "No bootstrap results produced. Check model and data paths."
        )

    raw = pd.DataFrame(all_rows)
    raw_path = RESULTS / "bootstrap_paired_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    group_columns = [
        "typology_group", "alpha", "markov_order"
    ]
    summary = raw.groupby(group_columns, as_index=False).agg(
        languages=("code", "nunique"),
        mean_delta=("delta_bpe_minus_unigram", "mean"),
        median_delta=("delta_bpe_minus_unigram", "median"),
        mean_language_ci_low=("bootstrap_ci_2_5", "mean"),
        mean_language_ci_high=("bootstrap_ci_97_5", "mean"),
        unigram_interval_wins=(
            "ci_interpretation",
            lambda s: int((s == "unigram_lower_cross_entropy").sum())
        ),
        bpe_interval_wins=(
            "ci_interpretation",
            lambda s: int((s == "bpe_lower_cross_entropy").sum())
        ),
        intervals_include_zero=(
            "ci_interpretation",
            lambda s: int((s == "interval_includes_zero").sum())
        )
    )
    summary_path = RESULTS / "bootstrap_paired_typology_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    make_plots(raw)

    print("\nBootstrap analysis completed.")
    print(f"Raw results: {raw_path}")
    print(f"Typology summary: {summary_path}")
    print(f"Bootstrap repetitions per comparison: {N_BOOTSTRAP}")
    print(f"Alphas: {ALPHAS}")
    print(f"Primary alpha for plots: {PLOT_ALPHA}")


if __name__ == "__main__":
    main()
