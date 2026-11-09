"""
Alpha sensitivity analysis for the empirical Markov n-gram model.

Reuses existing SentencePiece models and the saved training samples.
Does NOT train or overwrite tokenizers.

Run from the project root:
    python src/11_alpha_sensitivity.py
"""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2

from common import ROOT, CONFIG


# ============================================================
# SETTINGS
# ============================================================

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
TRAINING_SAMPLES = RESULTS / "training_samples"

# The original core experiment used alpha=0.1.
ALPHAS = [0.01, 0.1, 0.2, 0.5, 1.0]


# ============================================================
# DATA HELPERS
# ============================================================

def read_sentences(path, limit=None):
    """Read non-empty lines from a UTF-8 text file."""
    sentences = []

    with open(path, encoding="utf-8") as file:
        for line in file:
            sentence = line.strip()
            if sentence:
                sentences.append(sentence)

            if limit is not None and len(sentences) >= limit:
                break

    return sentences


def encode_sentences(processor, sentences):
    """Encode each sentence independently; sentence boundaries are preserved."""
    return [
        processor.encode(sentence, out_type=int)
        for sentence in sentences
    ]


def calculate_sample_hash(sentences):
    """Reproduce the SHA256 convention used by 03_train_and_analyze.py."""
    sample_text = "\n".join(sentences)
    return hashlib.sha256(sample_text.encode("utf-8")).hexdigest()


def get_emittable_vocab_size(processor):
    """Count NORMAL and UNKNOWN pieces, matching the core experiment."""
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(processor.serialized_model_proto())

    size = sum(
        1 for piece in proto.pieces
        if piece.type in (1, 2)
    )

    if size < 2:
        raise ValueError(f"Invalid emittable vocabulary size: {size}")

    return size


# ============================================================
# EMPIRICAL MARKOV MODEL
# These functions intentionally reproduce the implementation
# in 03_train_and_analyze.py so the alpha comparison is paired.
# ============================================================

def fit_ngram_counts(sequences, max_order, max_tokens, vocab_size):
    """
    Estimate empirical n-gram counts.

    Sentence boundaries are respected: contexts never cross
    from one sentence into another.
    """
    if max_order < 0:
        raise ValueError("max_order must be non-negative.")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    counts = [Counter() for _ in range(max_order + 1)]
    context_counts = [Counter() for _ in range(max_order + 1)]

    seen = 0

    for sequence in sequences:
        for i, token in enumerate(sequence):
            if seen >= max_tokens:
                break

            counts[0][(token,)] += 1

            for order in range(1, max_order + 1):
                if i >= order:
                    context = tuple(sequence[i - order:i])
                    counts[order][(context, token)] += 1
                    context_counts[order][context] += 1

            seen += 1

        if seen >= max_tokens:
            break

    return counts, context_counts, max(vocab_size, 2)


def eval_markov_cross_entropy(
    sequences,
    counts,
    context_counts,
    vocab_size,
    max_order,
    alpha,
):
    """
    Calculate held-out cross-entropy for orders 0..max_order.

    Uses add-alpha smoothing and the same variable-context
    convention at sentence beginnings as the core experiment.
    """
    if alpha <= 0:
        raise ValueError("Smoothing alpha must be positive.")

    output = []
    unigram_total = sum(counts[0].values())

    for order in range(max_order + 1):
        total_nll = 0.0
        n_tokens = 0

        for sequence in sequences:
            for i, token in enumerate(sequence):
                if order == 0:
                    observed = counts[0].get((token,), 0)
                    denominator = unigram_total + alpha * vocab_size
                else:
                    context = tuple(sequence[max(0, i - order):i])
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
                        denominator = (
                            context_total + alpha * vocab_size
                        )

                probability = (observed + alpha) / denominator

                if not 0 < probability <= 1:
                    raise ValueError(
                        f"Invalid probability {probability} for alpha={alpha}"
                    )

                total_nll -= math.log2(probability)
                n_tokens += 1

        if n_tokens == 0:
            raise ValueError("No evaluation tokens.")

        output.append({
            "markov_order": order,
            "cross_entropy_bits_per_token": total_nll / n_tokens,
            "total_nll_bits": total_nll,
            "n_eval_tokens": n_tokens,
        })

    return output


# ============================================================
# PLOTS
# ============================================================

def create_plots(raw):
    """Save plots of alpha sensitivity and paired BPE-Unigram differences."""

    grouped = (
        raw.groupby(
            ["algorithm", "alpha", "markov_order"],
            as_index=False,
        )["cross_entropy_bits_per_utf8_byte"]
        .mean()
    )

    for algorithm in sorted(grouped["algorithm"].unique()):
        subset = grouped[grouped["algorithm"] == algorithm]

        plt.figure(figsize=(9, 6))

        for alpha in ALPHAS:
            curve = subset[
                subset["alpha"] == alpha
            ].sort_values("markov_order")

            if not curve.empty:
                plt.plot(
                    curve["markov_order"],
                    curve["cross_entropy_bits_per_utf8_byte"],
                    marker="o",
                    label=f"alpha={alpha:g}",
                )

        plt.xlabel("Markov order")
        plt.ylabel("Cross-entropy (bits / UTF-8 byte)")
        plt.title(f"Alpha sensitivity — {algorithm.upper()}")
        plt.xticks(sorted(raw["markov_order"].unique()))
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            RESULTS / f"alpha_sensitivity_{algorithm}.png",
            dpi=180,
        )
        plt.close()

    # Paired difference: positive means Unigram has higher CE,
    # negative means Unigram has lower CE than BPE.
    paired = raw.pivot_table(
        index=[
            "code",
            "seed",
            "vocab_size_requested",
            "alpha",
            "markov_order",
        ],
        columns="algorithm",
        values="cross_entropy_bits_per_utf8_byte",
        aggfunc="first",
    ).reset_index()

    if {"bpe", "unigram"}.issubset(paired.columns):
        paired["unigram_minus_bpe"] = paired["unigram"] - paired["bpe"]

        diff = (
            paired.groupby(
                ["alpha", "markov_order"],
                as_index=False,
            )["unigram_minus_bpe"]
            .mean()
        )

        plt.figure(figsize=(9, 6))

        for alpha in ALPHAS:
            curve = diff[
                diff["alpha"] == alpha
            ].sort_values("markov_order")

            if not curve.empty:
                plt.plot(
                    curve["markov_order"],
                    curve["unigram_minus_bpe"],
                    marker="o",
                    label=f"alpha={alpha:g}",
                )

        plt.axhline(0, linewidth=1)
        plt.xlabel("Markov order")
        plt.ylabel("Unigram − BPE (bits / UTF-8 byte)")
        plt.title("Paired difference in cross-entropy")
        plt.xticks(sorted(raw["markov_order"].unique()))
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            RESULTS / "alpha_sensitivity_bpe_vs_unigram.png",
            dpi=180,
        )
        plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    manifest_path = ROOT / "data" / "prepared_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            "Cannot find data/prepared_manifest.json. "
            "Run src/02_prepare_data.py first."
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    experiment = CONFIG["experiment"]

    algorithms = [item.lower() for item in experiment["algorithms"]]
    vocab_sizes = experiment["vocab_sizes"]
    seeds = experiment["seeds"]

    max_order = experiment["max_markov_order"]
    max_train_tokens = experiment["max_ngram_train_tokens"]
    max_eval_sentences = experiment["max_eval_sentences"]

    RESULTS.mkdir(parents=True, exist_ok=True)

    # If the original raw results exist, use their sample hashes
    # to verify that the saved training sample is the one used
    # for the core experiment.
    original_metrics_path = RESULTS / "metrics_raw.csv"
    original_hashes = {}

    if original_metrics_path.exists():
        original_metrics = pd.read_csv(original_metrics_path)
        required = {
            "code", "algorithm", "vocab_size_requested",
            "seed", "training_sample_sha256",
        }

        if required.issubset(original_metrics.columns):
            original_hashes = (
                original_metrics.drop_duplicates(
                    ["code", "algorithm", "vocab_size_requested", "seed"]
                )
                .set_index(
                    ["code", "algorithm", "vocab_size_requested", "seed"]
                )["training_sample_sha256"]
                .to_dict()
            )

    rows = []
    skipped = []

    for code, meta in manifest["languages"].items():
        print(f"\n=== {meta['name']} ({code}) ===")

        test_file = ROOT / meta["test_file"]
        test_sentences = read_sentences(
            test_file,
            limit=max_eval_sentences,
        )

        if not test_sentences:
            print("  Skipping: no test sentences.")
            skipped.append((code, "empty test corpus"))
            continue

        total_utf8_bytes = sum(
            len(sentence.encode("utf-8"))
            for sentence in test_sentences
        )

        for algorithm in algorithms:
            if algorithm not in {"bpe", "unigram"}:
                raise ValueError(
                    f"Unsupported algorithm: {algorithm}"
                )

            for vocab_size in vocab_sizes:
                for seed in seeds:
                    model_name = (
                        f"{code}_{algorithm}_v{vocab_size}_s{seed}"
                    )
                    model_file = MODELS / f"{model_name}.model"
                    sample_file = (
                        TRAINING_SAMPLES / f"{code}_train_s{seed}.txt"
                    )

                    if not model_file.exists():
                        reason = f"missing model {model_file.name}"
                        print(f"  Skipping {model_name}: {reason}")
                        skipped.append((model_name, reason))
                        continue

                    if not sample_file.exists():
                        reason = f"missing sample {sample_file.name}"
                        print(f"  Skipping {model_name}: {reason}")
                        skipped.append((model_name, reason))
                        continue

                    # Read the complete saved sample for hash validation.
                    # The core experiment then used only its first 20,000
                    # sentences for Markov training.
                    saved_training_sample = read_sentences(sample_file)

                    if not saved_training_sample:
                        reason = "empty training sample"
                        print(f"  Skipping {model_name}: {reason}")
                        skipped.append((model_name, reason))
                        continue

                    sample_hash = calculate_sample_hash(
                        saved_training_sample
                    )
                    training_sentences = saved_training_sample[:20000]

                    hash_key = (
                        code, algorithm, vocab_size, seed
                    )

                    expected_hash = original_hashes.get(hash_key)
                    if (
                        expected_hash is not None
                        and str(expected_hash) != sample_hash
                    ):
                        raise RuntimeError(
                            f"Training sample hash mismatch for {model_name}.\n"
                            "The saved sample file differs from the sample "
                            "recorded in metrics_raw.csv. Do not interpret "
                            "the sensitivity results until this is resolved."
                        )

                    print(f"  Loading {model_name}")

                    processor = spm.SentencePieceProcessor(
                        model_file=str(model_file)
                    )

                    train_sequences = encode_sentences(
                        processor,
                        training_sentences,
                    )
                    test_sequences = encode_sentences(
                        processor,
                        test_sentences,
                    )

                    emittable_vocab_size = get_emittable_vocab_size(
                        processor
                    )

                    # N-gram counts are calculated only once per tokenizer.
                    # Alpha changes only the smoothing probabilities.
                    counts, context_counts, effective_vocab_size = (
                        fit_ngram_counts(
                            train_sequences,
                            max_order,
                            max_train_tokens,
                            emittable_vocab_size,
                        )
                    )

                    for alpha in ALPHAS:
                        ce_results = eval_markov_cross_entropy(
                            test_sequences,
                            counts,
                            context_counts,
                            effective_vocab_size,
                            max_order,
                            alpha,
                        )

                        for item in ce_results:
                            rows.append({
                                "language": meta["name"],
                                "code": code,
                                "typology_group": meta["group"],
                                "algorithm": algorithm,
                                "vocab_size_requested": vocab_size,
                                "vocab_size_actual": processor.get_piece_size(),
                                "emittable_vocab_size": emittable_vocab_size,
                                "seed": seed,
                                "training_sample_sha256": sample_hash,
                                "alpha": alpha,
                                "markov_order": item["markov_order"],
                                "cross_entropy_bits_per_token":
                                    item["cross_entropy_bits_per_token"],
                                "cross_entropy_bits_per_utf8_byte":
                                    item["total_nll_bits"]
                                    / max(total_utf8_bytes, 1),
                                "total_nll_bits": item["total_nll_bits"],
                                "n_eval_tokens": item["n_eval_tokens"],
                                "n_eval_sentences": len(test_sentences),
                                "n_eval_utf8_bytes": total_utf8_bytes,
                                "n_training_sentences": len(training_sentences),
                                "max_ngram_train_tokens": max_train_tokens,
                            })

                    print(
                        f"    Done: {len(ALPHAS)} alpha values, "
                        f"Markov orders 0–{max_order}"
                    )

    if not rows:
        raise SystemExit(
            "No results produced. Check model files, training samples "
            "and prepared_manifest.json."
        )

    raw = pd.DataFrame(rows)
    raw_path = RESULTS / "alpha_sensitivity_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    summary = (
        raw.groupby(
            ["algorithm", "alpha", "markov_order"],
            as_index=False,
        )
        .agg(
            mean_cross_entropy_bits_per_token=(
                "cross_entropy_bits_per_token", "mean"
            ),
            sd_cross_entropy_bits_per_token=(
                "cross_entropy_bits_per_token", "std"
            ),
            mean_cross_entropy_bits_per_utf8_byte=(
                "cross_entropy_bits_per_utf8_byte", "mean"
            ),
            sd_cross_entropy_bits_per_utf8_byte=(
                "cross_entropy_bits_per_utf8_byte", "std"
            ),
            n_models=("code", "count"),
        )
    )

    summary_path = RESULTS / "alpha_sensitivity_summary.csv"
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    by_language = (
        raw.groupby(
            [
                "code",
                "language",
                "typology_group",
                "algorithm",
                "alpha",
                "markov_order",
            ],
            as_index=False,
        )
        .agg(
            cross_entropy_bits_per_token=(
                "cross_entropy_bits_per_token", "mean"
            ),
            cross_entropy_bits_per_utf8_byte=(
                "cross_entropy_bits_per_utf8_byte", "mean"
            ),
        )
    )

    by_language_path = RESULTS / "alpha_sensitivity_by_language.csv"
    by_language.to_csv(
        by_language_path,
        index=False,
        encoding="utf-8-sig",
    )

    create_plots(raw)

    print("\nAlpha sensitivity analysis completed.")
    print(f"Raw results:      {raw_path}")
    print(f"Overall summary:  {summary_path}")
    print(f"By-language:      {by_language_path}")
    print("Plots:")
    print(RESULTS / "alpha_sensitivity_bpe.png")
    print(RESULTS / "alpha_sensitivity_unigram.png")
    print(RESULTS / "alpha_sensitivity_bpe_vs_unigram.png")

    if skipped:
        print("\nSkipped models:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    print(
        "\nNo SentencePiece models were trained or overwritten."
    )


if __name__ == "__main__":
    main()
