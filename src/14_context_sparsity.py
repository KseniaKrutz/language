"""
14_context_sparsity.py

Diagnoses sparsity of Markov contexts for SentencePiece BPE and Unigram.

Uses the same trained models, saved training samples, held-out test sentences,
and token-count cap as the main experiment.

Main questions:
1. How many evaluation tokens have a context observed in training?
2. How often do contexts occur in the training sample?
3. Does context sparsity increase with Markov order?
4. How much cross-entropy is contributed by rare/unseen contexts?
5. Do BPE and Unigram differ in context coverage?

Important:
- Sentence boundaries are respected.
- At the beginning of a sentence, the effective context is shorter, matching
  the main experiment's variable-length context handling.
- For k=0 there is no preceding context; this is reported separately.
- "Context frequency" means the number of training occurrences of the
  effective preceding-token sequence, not the frequency of the target token.
- This is a diagnostic of the fitted finite-sample Markov model, not a claim
  that natural language itself is Markovian.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm

from common import ROOT, CONFIG


RESULTS = ROOT / "results"
MODELS = ROOT / "models"
TRAINING_SAMPLES = RESULTS / "training_samples"

MAX_TRAIN_SENTENCES = 20000
FREQUENCY_BINS = [
    ("unseen_0", 0, 0),
    ("rare_1", 1, 1),
    ("low_2_5", 2, 5),
    ("medium_6_20", 6, 20),
    ("frequent_21_plus", 21, None),
]


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


def encode_sentences(sp, sentences):
    return [sp.encode(sentence, out_type=int) for sentence in sentences]


def fit_context_counts(train_sequences, max_order, max_tokens):
    """
    Count effective contexts of lengths 0..max_order.

    For each token at position i, the context for order k is the last
    min(k, i) tokens in the same sentence. Counting stops after max_tokens
    target tokens, as in the main Markov experiment.
    """
    context_counts = [Counter() for _ in range(max_order + 1)]
    token_counts = [Counter() for _ in range(max_order + 1)]
    used_tokens = 0

    for sequence in train_sequences:
        for i, target in enumerate(sequence):
            if used_tokens >= max_tokens:
                break

            # Unigram counts (order 0).
            context_counts[0][()] += 1
            token_counts[0][((), target)] += 1

            # For order k, count only full k-token contexts.
            # At sentence beginnings, evaluation backs off to the matching
            # lower-order table, as in the main experiment.
            for k in range(1, max_order + 1):
                if i >= k:
                    context = tuple(sequence[i-k:i])
                    context_counts[k][context] += 1
                    token_counts[k][(context, target)] += 1

            used_tokens += 1

        if used_tokens >= max_tokens:
            break

    return context_counts, token_counts, used_tokens


def frequency_bin(freq):
    if freq == 0:
        return "unseen_0"
    if freq == 1:
        return "rare_1"
    if freq <= 5:
        return "low_2_5"
    if freq <= 20:
        return "medium_6_20"
    return "frequent_21_plus"


def evaluate_contexts(test_sequences, context_counts, token_counts,
                      vocab_size, max_order, alpha, byte_lengths):
    """
    Produce per-language/model/order summaries and frequency-bin summaries.

    Cross-entropy contribution is normalized by total UTF-8 bytes in the
    test corpus, so bin contributions add up to the overall bits/byte.
    """
    total_bytes = float(np.sum(byte_lengths))
    total_eval_tokens = sum(len(seq) for seq in test_sequences)

    overall_rows = []
    bin_rows = []

    for k in range(max_order + 1):
        total_nll = 0.0
        observed_context_tokens = 0
        unseen_context_tokens = 0
        context_frequency_sum = 0
        distinct_eval_contexts = set()
        distinct_unseen_contexts = set()

        # Accumulate token counts, context counts, NLL, and byte-weighted
        # contributions by context-frequency bucket.
        bucket = defaultdict(lambda: {
            "tokens": 0,
            "context_frequency_sum": 0,
            "nll_bits": 0.0,
            "distinct_contexts": set()
        })

        for sent_idx, sequence in enumerate(test_sequences):
            for i, target in enumerate(sequence):
                effective_order = min(k, i)
                context = tuple(sequence[i-effective_order:i])

                # Use the actual effective-order table at sentence starts.
                context_freq = context_counts[effective_order].get(context, 0)
                target_count = token_counts[effective_order].get(
                    (context, target), 0
                )

                denominator = context_freq + alpha * vocab_size
                probability = (target_count + alpha) / denominator
                nll = -np.log2(probability)

                total_nll += nll
                context_frequency_sum += context_freq
                distinct_eval_contexts.add(context)

                if context_freq > 0:
                    observed_context_tokens += 1
                else:
                    unseen_context_tokens += 1
                    distinct_unseen_contexts.add(context)

                label = frequency_bin(context_freq)
                bucket[label]["tokens"] += 1
                bucket[label]["context_frequency_sum"] += context_freq
                bucket[label]["nll_bits"] += nll
                bucket[label]["distinct_contexts"].add(context)

        total_bits_per_byte = total_nll / total_bytes if total_bytes else np.nan
        overall_rows.append({
            "markov_order": k,
            "alpha": alpha,
            "eval_tokens": total_eval_tokens,
            "test_utf8_bytes": int(total_bytes),
            "observed_context_token_share": (
                observed_context_tokens / total_eval_tokens
                if total_eval_tokens else np.nan
            ),
            "unseen_context_token_share": (
                unseen_context_tokens / total_eval_tokens
                if total_eval_tokens else np.nan
            ),
            "mean_training_context_frequency_per_eval_token": (
                context_frequency_sum / total_eval_tokens
                if total_eval_tokens else np.nan
            ),
            "distinct_eval_contexts": len(distinct_eval_contexts),
            "distinct_unseen_eval_contexts": len(distinct_unseen_contexts),
            "cross_entropy_bits_per_utf8_byte": total_bits_per_byte,
            "cross_entropy_bits_per_token": (
                total_nll / total_eval_tokens if total_eval_tokens else np.nan
            )
        })

        for label, _, _ in FREQUENCY_BINS:
            values = bucket[label]
            n_tokens = values["tokens"]
            bin_rows.append({
                "markov_order": k,
                "alpha": alpha,
                "context_frequency_bin": label,
                "eval_tokens": n_tokens,
                "eval_token_share": (
                    n_tokens / total_eval_tokens if total_eval_tokens else np.nan
                ),
                "mean_training_context_frequency": (
                    values["context_frequency_sum"] / n_tokens
                    if n_tokens else np.nan
                ),
                "distinct_eval_contexts": len(values["distinct_contexts"]),
                "nll_bits": values["nll_bits"],
                "cross_entropy_contribution_bits_per_utf8_byte": (
                    values["nll_bits"] / total_bytes if total_bytes else np.nan
                ),
                "mean_nll_bits_per_token": (
                    values["nll_bits"] / n_tokens if n_tokens else np.nan
                )
            })

    return overall_rows, bin_rows


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / "data" / "prepared_manifest.json"
    metrics_path = RESULTS / "metrics_raw.csv"

    if not manifest_path.exists():
        raise SystemExit(f"Missing prepared manifest: {manifest_path}")
    if not metrics_path.exists():
        raise SystemExit(f"Missing core results: {metrics_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)

    required = {"code", "seed", "vocab_size_requested", "algorithm"}
    missing = required - set(metrics.columns)
    if missing:
        raise SystemExit(
            f"metrics_raw.csv is missing columns: {sorted(missing)}"
        )

    # Process only combinations present in the completed core experiment.
    combinations = (
        metrics[["code", "seed", "vocab_size_requested"]]
        .drop_duplicates()
        .sort_values(["code", "seed", "vocab_size_requested"])
    )

    experiment = CONFIG["experiment"]
    max_order = int(experiment["max_markov_order"])
    max_train_tokens = int(experiment["max_ngram_train_tokens"])
    max_eval_sentences = int(experiment["max_eval_sentences"])
    alpha = float(experiment["smoothing_alpha"])

    overall_all = []
    bins_all = []

    for item in combinations.itertuples(index=False):
        code = str(item.code)
        seed = int(item.seed)
        vocab_requested = int(item.vocab_size_requested)

        if code not in manifest["languages"]:
            print(f"Skipping {code}: not present in prepared manifest.")
            continue

        meta = manifest["languages"][code]
        test_path = ROOT / meta["test_file"]
        train_path = TRAINING_SAMPLES / f"{code}_train_s{seed}.txt"

        if not test_path.exists():
            print(f"Skipping {code}: test file not found: {test_path}")
            continue
        if not train_path.exists():
            print(f"Skipping {code}: training sample not found: {train_path}")
            continue

        train_sentences = read_sentences(
            train_path, MAX_TRAIN_SENTENCES
        )
        test_sentences = read_sentences(
            test_path, max_eval_sentences
        )

        if not train_sentences or not test_sentences:
            print(f"Skipping {code}: empty train or test corpus.")
            continue

        byte_lengths = np.array(
            [len(s.encode("utf-8")) for s in test_sentences],
            dtype=float
        )

        print(
            f"\n=== {meta['name']} ({code}), seed={seed}, "
            f"vocab={vocab_requested} ==="
        )

        for algorithm in ("bpe", "unigram"):
            model_path = (
                MODELS
                / f"{code}_{algorithm}_v{vocab_requested}_s{seed}.model"
            )
            if not model_path.exists():
                print(f"Skipping {algorithm}: model not found: {model_path}")
                continue

            sp = spm.SentencePieceProcessor(model_file=str(model_path))
            vocab_size = sp.get_piece_size()

            train_sequences = encode_sentences(sp, train_sentences)
            test_sequences = encode_sentences(sp, test_sentences)

            context_counts, token_counts, used_train_tokens = (
                fit_context_counts(
                    train_sequences,
                    max_order,
                    max_train_tokens
                )
            )

            overall_rows, bin_rows = evaluate_contexts(
                test_sequences,
                context_counts,
                token_counts,
                vocab_size,
                max_order,
                alpha,
                byte_lengths
            )

            for row in overall_rows:
                row.update({
                    "language": meta["name"],
                    "code": code,
                    "typology_group": meta["group"],
                    "algorithm": algorithm,
                    "seed": seed,
                    "vocab_size_requested": vocab_requested,
                    "vocab_size_actual": vocab_size,
                    "alpha": alpha,
                    "train_tokens_used": used_train_tokens,
                    "train_sentences_loaded": len(train_sentences),
                    "test_sentences": len(test_sentences)
                })
                overall_all.append(row)

            for row in bin_rows:
                row.update({
                    "language": meta["name"],
                    "code": code,
                    "typology_group": meta["group"],
                    "algorithm": algorithm,
                    "seed": seed,
                    "vocab_size_requested": vocab_requested,
                    "vocab_size_actual": vocab_size,
                    "train_tokens_used": used_train_tokens
                })
                bins_all.append(row)

            print(
                f"  {algorithm}: vocab={vocab_size}, "
                f"train tokens used={used_train_tokens}"
            )

    if not overall_all:
        raise SystemExit(
            "No results produced. Check model, corpus, and sample paths."
        )

    overall = pd.DataFrame(overall_all)
    bins = pd.DataFrame(bins_all)

    overall_path = RESULTS / "context_sparsity_raw.csv"
    bins_path = RESULTS / "context_frequency_bins_raw.csv"

    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    bins.to_csv(bins_path, index=False, encoding="utf-8-sig")

    # Algorithm-level summary by typology, order, and alpha.
    group_summary = overall.groupby(
        ["typology_group", "algorithm", "markov_order", "alpha"],
        as_index=False
    ).agg(
        languages=("code", "nunique"),
        mean_observed_context_share=("observed_context_token_share", "mean"),
        mean_unseen_context_share=("unseen_context_token_share", "mean"),
        mean_context_frequency=("mean_training_context_frequency_per_eval_token", "mean"),
        mean_cross_entropy_bits_per_byte=(
            "cross_entropy_bits_per_utf8_byte", "mean"
        )
    )
    group_path = RESULTS / "context_sparsity_typology_summary.csv"
    group_summary.to_csv(group_path, index=False, encoding="utf-8-sig")

    # Paired BPE-Unigram differences within each language.
    index_cols = [
        "language", "code", "typology_group", "seed",
        "vocab_size_requested", "markov_order", "alpha"
    ]
    wide = overall.pivot_table(
        index=index_cols,
        columns="algorithm",
        values=[
            "observed_context_token_share",
            "unseen_context_token_share",
            "mean_training_context_frequency_per_eval_token",
            "cross_entropy_bits_per_utf8_byte"
        ],
        aggfunc="first"
    )
    wide.columns = [
        f"{metric}_{algorithm}" for metric, algorithm in wide.columns
    ]
    wide = wide.reset_index()

    for metric in [
        "observed_context_token_share",
        "unseen_context_token_share",
        "mean_training_context_frequency_per_eval_token",
        "cross_entropy_bits_per_utf8_byte"
    ]:
        bpe_col = f"{metric}_bpe"
        unigram_col = f"{metric}_unigram"
        if bpe_col in wide.columns and unigram_col in wide.columns:
            wide[f"delta_bpe_minus_unigram_{metric}"] = (
                wide[bpe_col] - wide[unigram_col]
            )

    paired_path = RESULTS / "context_sparsity_bpe_vs_unigram.csv"
    wide.to_csv(paired_path, index=False, encoding="utf-8-sig")

    # Compact frequency-bin summary.
    bin_summary = bins.groupby(
        ["algorithm", "context_frequency_bin", "markov_order"],
        as_index=False
    ).agg(
        language_rows=("code", "nunique"),
        mean_eval_token_share=("eval_token_share", "mean"),
        mean_cross_entropy_contribution_bits_per_byte=(
            "cross_entropy_contribution_bits_per_utf8_byte", "mean"
        ),
        mean_nll_bits_per_token=("mean_nll_bits_per_token", "mean")
    )
    bin_summary_path = RESULTS / "context_frequency_bins_summary.csv"
    bin_summary.to_csv(
        bin_summary_path, index=False, encoding="utf-8-sig"
    )

    print("\nContext sparsity analysis completed.")
    print(f"Overall metrics: {overall_path}")
    print(f"Frequency bins: {bins_path}")
    print(f"Typology summary: {group_path}")
    print(f"BPE vs Unigram: {paired_path}")
    print(f"Frequency-bin summary: {bin_summary_path}")

    # Optional plots; CSV outputs remain the primary results.
    try:
        import matplotlib.pyplot as plt

        for algorithm in ("bpe", "unigram"):
            subset = overall[overall["algorithm"] == algorithm]
            plot_data = subset.groupby("markov_order", as_index=False).agg(
                unseen_share=("unseen_context_token_share", "mean"),
                entropy=("cross_entropy_bits_per_utf8_byte", "mean")
            )

            plt.figure(figsize=(8, 5))
            plt.plot(
                plot_data["markov_order"],
                plot_data["unseen_share"],
                marker="o"
            )
            plt.xlabel("Markov order")
            plt.ylabel("Share of test tokens with unseen context")
            plt.ylim(0, 1)
            plt.title(f"Unseen context share — {algorithm.upper()}")
            plt.tight_layout()
            plt.savefig(
                RESULTS / f"context_unseen_share_{algorithm}.png",
                dpi=180
            )
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(
                plot_data["markov_order"],
                plot_data["entropy"],
                marker="o"
            )
            plt.xlabel("Markov order")
            plt.ylabel("Cross-entropy (bits / UTF-8 byte)")
            plt.title(f"Cross-entropy vs unseen contexts — {algorithm.upper()}")
            plt.tight_layout()
            plt.savefig(
                RESULTS / f"context_entropy_{algorithm}.png",
                dpi=180
            )
            plt.close()

    except ImportError:
        print("matplotlib not installed; skipping plots.")


if __name__ == "__main__":
    main()
