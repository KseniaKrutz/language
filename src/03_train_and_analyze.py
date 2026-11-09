
"""
Core experiment
---------------
1. Train SentencePiece BPE and Unigram tokenizers.
2. Measure tokenization fertility.
3. Estimate empirical Markov cross-entropy at different orders.
4. For Unigram, sample alternative segmentations and measure variability.
5. Save raw results, summaries, sampling diagnostics and plots.

The Markov model is an empirical n-gram approximation.
It does not assume that natural language is exactly a finite-order
Markov chain.

Experimental design:
- Different seeds define reproducible training subsets.
- The same subset is used for tokenizer training and Markov training.
- All models are evaluated on the same held-out test sentences.
"""

import random
import math
import json
import hashlib

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import sentencepiece as spm
import matplotlib.pyplot as plt

from sentencepiece import sentencepiece_model_pb2

from common import ROOT, CONFIG, write_json


# ============================================================
# PATHS AND EXPERIMENT SETTINGS
# ============================================================

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
TRAINING_SAMPLES = RESULTS / "training_samples"

MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
TRAINING_SAMPLES.mkdir(parents=True, exist_ok=True)

TRAIN_SAMPLE_FRACTION = 0.9

# True: always retrain models, replacing existing files.
# False: reuse existing models when their .model file exists.
RETRAIN_MODELS = True

# Maximum number of test sentences considered before token-budget filtering.
MAX_EVAL_SENTENCES = 5000


# ============================================================
# DATA HELPERS
# ============================================================

def read_sentences(path, limit=None):
    """Read non-empty lines from a UTF-8 text file."""
    result = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()

            if sentence:
                result.append(sentence)

            if limit is not None and len(result) >= limit:
                break

    return result


def make_seeded_training_sample(
    input_file,
    output_file,
    seed_value,
    fraction=TRAIN_SAMPLE_FRACTION
):
    """
    Create a reproducible random subset of the training corpus.

    The same seed produces the same subset.
    Different seeds produce different subsets, unless the corpus
    contains too few distinct sentences.
    """

    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be in (0, 1].")

    sentences = read_sentences(input_file)

    if not sentences:
        raise ValueError(f"Empty training corpus: {input_file}")

    rng = random.Random(seed_value)

    sample_size = max(1, int(len(sentences) * fraction))

    selected = rng.sample(sentences, sample_size)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in selected:
            f.write(sentence + "\n")

    sample_text = "\n".join(selected)
    sample_hash = hashlib.sha256(
        sample_text.encode("utf-8")
    ).hexdigest()

    return {
        "sentences": selected,
        "n_sentences": len(selected),
        "sha256": sample_hash,
        "path": str(output_file)
    }


def encode_sentences(sp, sentences):
    """Encode sentences into SentencePiece token IDs."""
    return [
        sp.encode(sentence, out_type=int)
        for sentence in sentences
    ]


def select_eval_subset(sp, sentences, max_sentences, max_tokens=None):
    """
    Use the same held-out sentences for every tokenizer.

    Selection depends only on sentence count, not on tokenization.
    This guarantees comparable evaluation samples across models.
    """

    selected_sentences = sentences[:max_sentences]
    selected_sequences = encode_sentences(sp, selected_sentences)

    if not selected_sentences:
        raise ValueError("No sentences selected for evaluation.")

    return selected_sentences, selected_sequences


# ============================================================
# SENTENCEPIECE TRAINING
# ============================================================

def train_spm(
    input_file,
    model_prefix,
    model_type,
    vocab_size,
    code
):
    """Train a SentencePiece tokenizer and return its model path."""

    cjk = code in {"zho", "jpn"}

    coverage = (
        CONFIG["experiment"]["sentencepiece_character_coverage_cjk"]
        if cjk
        else CONFIG["experiment"]["sentencepiece_character_coverage_latin"]
    )

    model_file = Path(str(model_prefix) + ".model")
    vocab_file = Path(str(model_prefix) + ".vocab")

    if model_file.exists() and not RETRAIN_MODELS:
        print(f"  Reusing existing model: {model_file}")
        return model_file

    for path in (model_file, vocab_file):
        if path.exists():
            path.unlink()

    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(model_prefix),
        model_type=model_type.lower(),
        vocab_size=vocab_size,
        character_coverage=coverage,
        input_sentence_size=0,
        shuffle_input_sentence=False,
        num_threads=1,
        hard_vocab_limit=False,
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
        user_defined_symbols=[],

        # Suppress verbose INFO messages from SentencePiece.
        minloglevel=2
    )

    if not model_file.exists():
        raise RuntimeError(
            f"SentencePiece did not create model: {model_file}"
        )

    return model_file

def get_emittable_vocab_size(sp):
    """
    Count token types that can appear in ordinary encoded sequences.

    NORMAL and UNKNOWN are included.
    CONTROL, UNUSED and other non-emittable special pieces are excluded.
    """

    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(sp.serialized_model_proto())

    vocab_size = sum(
        1
        for piece in proto.pieces
        if piece.type in (1, 2)  # NORMAL, UNKNOWN
    )

    if vocab_size < 2:
        raise ValueError(
            f"Invalid emittable vocabulary size: {vocab_size}"
        )

    return vocab_size


# ============================================================
# TOKENIZATION METRICS
# ============================================================

def token_metrics(sp, sentences, sequences=None):
    """Calculate tokenization fertility and corpus statistics."""

    if sequences is None:
        sequences = encode_sentences(sp, sentences)

    token_count = sum(len(ids) for ids in sequences)

    char_count = sum(len(sentence) for sentence in sentences)

    utf8_bytes = sum(
        len(sentence.encode("utf-8"))
        for sentence in sentences
    )

    whitespace_units = sum(
        len(sentence.split())
        for sentence in sentences
    )

    return {
        "n_sentences": len(sentences),
        "n_tokens": token_count,
        "n_unicode_chars": char_count,
        "n_utf8_bytes": utf8_bytes,
        "n_whitespace_units": whitespace_units,

        "tokens_per_unicode_char":
            token_count / max(char_count, 1),

        "tokens_per_utf8_byte":
            token_count / max(utf8_bytes, 1),

        "tokens_per_whitespace_unit":
            token_count / max(whitespace_units, 1)
    }


# ============================================================
# EMPIRICAL MARKOV MODEL
# ============================================================

def fit_ngram_counts(
    sequences,
    max_order,
    max_tokens,
    vocab_size
):
    """
    Estimate empirical n-gram counts.

    Sentence boundaries are respected: contexts never cross
    from one sentence into another.
    """

    if max_order < 0:
        raise ValueError("max_order must be non-negative.")

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    counts = [
        Counter()
        for _ in range(max_order + 1)
    ]

    context_counts = [
        Counter()
        for _ in range(max_order + 1)
    ]

    seen = 0

    for sequence in sequences:

        for i, token in enumerate(sequence):

            if seen >= max_tokens:
                break

            # Unigram counts.
            counts[0][(token,)] += 1

            # Higher-order n-gram counts.
            for k in range(1, max_order + 1):

                if i >= k:

                    context = tuple(sequence[i-k:i])

                    counts[k][(context, token)] += 1
                    context_counts[k][context] += 1

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
    alpha
):
    """
    Calculate held-out cross-entropy for orders 0 ... max_order.

    Add-alpha smoothing:

        P(token | context) =
        (count(context, token) + alpha)
        /
        (count(context) + alpha * V)

    At sentence beginnings, use the longest available shorter
    context. This is a variable-context approximation.
    """

    if alpha <= 0:
        raise ValueError("Smoothing alpha must be positive.")

    output = []

    unigram_total = sum(counts[0].values())

    for k in range(max_order + 1):

        total_nll = 0.0
        n = 0

        for sequence in sequences:

            for i, token in enumerate(sequence):

                if k == 0:

                    observed = counts[0].get((token,), 0)

                    denominator = (
                        unigram_total + alpha * vocab_size
                    )

                else:

                    context = tuple(sequence[max(0, i-k):i])

                    # At sentence beginnings, use shorter contexts.
                    effective_order = len(context)

                    if effective_order == 0:

                        observed = counts[0].get((token,), 0)

                        denominator = (
                            unigram_total + alpha * vocab_size
                        )

                    else:

                        observed = counts[effective_order].get(
                            (context, token), 0
                        )

                        context_total = context_counts[
                            effective_order
                        ].get(context, 0)

                        denominator = (
                            context_total + alpha * vocab_size
                        )

                probability = (
                    (observed + alpha) / denominator
                )

                if not 0 < probability <= 1:
                    raise ValueError(
                        f"Invalid probability: {probability}"
                    )

                total_nll -= math.log2(probability)
                n += 1

        if n == 0:
            raise ValueError("No evaluation tokens.")

        output.append({
            "order": k,
            "cross_entropy_bits_per_token": total_nll / n,
            "total_nll_bits": total_nll,
            "n_eval_tokens": n
        })

    return output


# ============================================================
# UNIGRAM SAMPLING VARIABILITY
# ============================================================

def sample_variability(
    sp,
    sentences,
    repetitions,
    alpha,
    max_sentences=500
):
    """
    Sample alternative Unigram segmentations.

    Measures variation in the number of tokens per sentence.
    It does not measure boundary-level uncertainty.
    """

    if repetitions < 2:
        raise ValueError("Sampling repetitions must be at least 2.")

    if alpha <= 0:
        raise ValueError("Sampling alpha must be positive.")

    rows = []

    for sentence in sentences[:max_sentences]:

        sampled_lengths = []

        for _ in range(repetitions):

            ids = sp.encode(
                sentence,
                out_type=int,
                enable_sampling=True,
                nbest_size=-1,
                alpha=alpha
            )

            sampled_lengths.append(len(ids))

        mean_length = float(np.mean(sampled_lengths))

        sd_length = float(
            np.std(sampled_lengths, ddof=1)
        )

        rows.append({
            "mean": mean_length,
            "sd": sd_length
        })

    return {
        "sampled_sentences": len(rows),

        "mean_tokens_per_sentence_across_samples": (
            float(np.mean([row["mean"] for row in rows]))
            if rows else None
        ),

        "mean_within_sentence_token_count_sd": (
            float(np.mean([row["sd"] for row in rows]))
            if rows else None
        )
    }


# ============================================================
# PLOTTING
# ============================================================

def create_plots(summary):
    """Create diagnostic plots."""

    # Plot 1: Markov cross-entropy by order.
    plt.figure(figsize=(13, 7))

    plot_df = summary[
        summary["markov_order"] > 0
    ]

    for (language, algorithm, vocab_size), group in plot_df.groupby(
        ["language", "algorithm", "vocab_size_requested"]
    ):

        group = group.sort_values("markov_order")

        plt.plot(
            group["markov_order"],
            group["mean_cross_entropy_bits_per_token"],
            marker="o",
            label=f"{language}-{algorithm}-{vocab_size}"
        )

    plt.xlabel("Markov order k")
    plt.ylabel("Held-out cross-entropy (bits/token)")
    plt.title("Markov cross-entropy by model order")
    plt.legend(fontsize=6, ncol=3)
    plt.tight_layout()

    plt.savefig(
        RESULTS / "markov_cross_entropy.png",
        dpi=180
    )

    plt.close()

    # Plot 2: Tokenization fertility.
    plot_df = summary[
        summary["markov_order"] == 1
    ].copy()

    plot_df = plot_df.sort_values(
        ["language", "algorithm", "vocab_size_requested"]
    )

    labels = [
        f"{row.language}\n{row.algorithm}\n{row.vocab_size_requested}"
        for row in plot_df.itertuples()
    ]

    plt.figure(
        figsize=(max(10, len(labels) * 0.35), 6)
    )

    plt.bar(
        range(len(labels)),
        plot_df["mean_tokens_per_unicode_char"]
    )

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=90,
        fontsize=7
    )

    plt.ylabel("Tokens per Unicode character")
    plt.title("Tokenization fertility")
    plt.tight_layout()

    plt.savefig(
        RESULTS / "token_fertility.png",
        dpi=180
    )

    plt.close()


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    manifest_path = ROOT / "data" / "prepared_manifest.json"

    if not manifest_path.exists():
        raise SystemExit(
            "Run first: python src/02_prepare_data.py"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    experiment_config = CONFIG["experiment"]

    algorithms = experiment_config["algorithms"]
    vocab_sizes = experiment_config["vocab_sizes"]
    seeds = experiment_config["seeds"]

    max_order = experiment_config["max_markov_order"]
    max_train_tokens = experiment_config["max_ngram_train_tokens"]
    max_eval_tokens = experiment_config["max_ngram_eval_tokens"]

    smoothing_alpha = experiment_config["smoothing_alpha"]

    sampling_repetitions = experiment_config["sampling_repetitions"]
    sampling_alpha = experiment_config["sampling_alpha"]

    max_eval_sentences = min(
        experiment_config["max_eval_sentences"],
        MAX_EVAL_SENTENCES
    )

    all_rows = []

    for code, meta in manifest["languages"].items():

        print(f"\n=== {meta['name']} ({code}) ===")

        train_file = ROOT / meta["train_file"]
        test_file = ROOT / meta["test_file"]

        train_sents = read_sentences(train_file)

        test_sents = read_sentences(
            test_file,
            experiment_config["max_eval_sentences"]
        )

        if len(train_sents) < 1000 or len(test_sents) < 100:

            print("Skipping: insufficient data.")
            continue

        # ----------------------------------------------------
        # Prepare one training subset per seed.
        # The same subset is reused for BPE, Unigram and
        # all vocabulary sizes.
        # ----------------------------------------------------

        samples_by_seed = {}

        for seed_value in seeds:

            sample_file = (
                TRAINING_SAMPLES
                / f"{code}_train_s{seed_value}.txt"
            )

            sample_info = make_seeded_training_sample(
                train_file,
                sample_file,
                seed_value,
                fraction=TRAIN_SAMPLE_FRACTION
            )

            samples_by_seed[seed_value] = sample_info

            print(
                f"  Seed {seed_value}: "
                f"{sample_info['n_sentences']} training sentences"
            )

        # ----------------------------------------------------
        # Train and evaluate models.
        # ----------------------------------------------------

        for algorithm in algorithms:

            algorithm = algorithm.lower()

            if algorithm not in {"bpe", "unigram"}:
                raise ValueError(
                    f"Unsupported SentencePiece algorithm: {algorithm}"
                )

            for vocab_size in vocab_sizes:

                for seed_value in seeds:

                    sample_info = samples_by_seed[seed_value]
                    sample_sents = sample_info["sentences"]
                    sample_file = sample_info["path"]

                    model_name = (
                        f"{code}_{algorithm}_v{vocab_size}_s{seed_value}"
                    )

                    model_prefix = MODELS / model_name

                    print(
                        f"\n  Training {model_name}"
                    )

                    model_file = train_spm(
                        sample_file,
                        model_prefix,
                        algorithm,
                        vocab_size,
                        code
                    )

                    model_file = str(model_prefix) + ".model"

                    sp = spm.SentencePieceProcessor(
                        model_file=model_file
                    )

                    actual_vocab_size = sp.get_piece_size()

                    emittable_vocab_size = get_emittable_vocab_size(sp)

                    # ------------------------------------------------
                    # Select a common held-out evaluation subset.
                    # All metrics use exactly these sentences.
                    # ------------------------------------------------

                    eval_subset, test_sequences = select_eval_subset(
                        sp,
                        test_sents,
                        max_eval_sentences,
                        max_eval_tokens
                    )

                    tm = token_metrics(
                        sp,
                        eval_subset,
                        test_sequences
                    )

                    # ------------------------------------------------
                    # Fit Markov model on the same seeded subset.
                    # ------------------------------------------------

                    # Explicitly retrieve the training sample for this seed.
                    sample_info = samples_by_seed[seed_value]
                    sample_sents = sample_info["sentences"]

                    if not sample_sents:
                        raise ValueError(
                            f"Empty training sample: language={code}, seed={seed_value}"
                        )

                    markov_train_sents = sample_sents[:20000]

                    markov_train_sequences = encode_sentences(
                        sp,
                        markov_train_sents
                    )

                    markov_train_sequences = encode_sentences(
                        sp,
                        markov_train_sents
                    )

                    counts, ctx_counts, vsize = fit_ngram_counts(
                        markov_train_sequences,
                        max_order,
                        max_train_tokens,
                        emittable_vocab_size
                    )

                    ce = eval_markov_cross_entropy(
                        test_sequences,
                        counts,
                        ctx_counts,
                        vsize,
                        max_order,
                        smoothing_alpha
                    )

                    # ------------------------------------------------
                    # Save results.
                    # ------------------------------------------------

                    for item in ce:

                        bits_per_byte = (
                            item["total_nll_bits"]
                            / max(tm["n_utf8_bytes"], 1)
                        )

                        all_rows.append({

                            "language": meta["name"],
                            "code": code,
                            "typology_group": meta["group"],

                            "algorithm": algorithm,

                            "vocab_size_requested": vocab_size,
                            "vocab_size_actual": actual_vocab_size,
                            "emittable_vocab_size": emittable_vocab_size,

                            "seed": seed_value,

                            "training_sample_fraction": TRAIN_SAMPLE_FRACTION,
                            "training_sample_sentences": sample_info["n_sentences"],
                            "training_sample_sha256": sample_info["sha256"],

                            **tm,

                            "markov_order": item["order"],

                            "cross_entropy_bits_per_token":
                                item["cross_entropy_bits_per_token"],

                            "cross_entropy_bits_per_utf8_byte":
                                bits_per_byte,

                            "n_eval_tokens": item["n_eval_tokens"],

                            "total_nll_bits": item["total_nll_bits"]
                        })

                    # ------------------------------------------------
                    # Unigram alternative segmentation sampling.
                    # ------------------------------------------------

                    if algorithm == "unigram":

                        sampling_result = sample_variability(
                            sp,
                            eval_subset,
                            sampling_repetitions,
                            sampling_alpha
                        )

                        write_json(
                            RESULTS / f"{model_name}_sampling.json",
                            sampling_result
                        )

                    print(
                        f"  {algorithm} vocab={vocab_size} "
                        f"seed={seed_value}: done"
                    )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    if not all_rows:

        raise SystemExit(
            "No experiment rows produced. Check corpus and configuration."
        )

    df = pd.DataFrame(all_rows)

    raw_path = RESULTS / "metrics_raw.csv"

    df.to_csv(
        raw_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # AGGREGATE RESULTS ACROSS SEEDS
    # ========================================================

    group_cols = [
        "language",
        "code",
        "typology_group",
        "algorithm",
        "vocab_size_requested",
        "markov_order"
    ]

    summary = df.groupby(
        group_cols,
        dropna=False
    ).agg(

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

        mean_tokens_per_unicode_char=(
            "tokens_per_unicode_char", "mean"
        ),

        sd_tokens_per_unicode_char=(
            "tokens_per_unicode_char", "std"
        ),

        mean_tokens_per_utf8_byte=(
            "tokens_per_utf8_byte", "mean"
        ),

        mean_tokens_per_whitespace_unit=(
            "tokens_per_whitespace_unit", "mean"
        ),

        mean_actual_vocab=(
            "vocab_size_actual", "mean"
        )

    ).reset_index()

    summary_path = RESULTS / "metrics_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # DIAGNOSTIC PLOTS
    # ========================================================

    create_plots(summary)

    # ========================================================
    # FINISH
    # ========================================================

    print("\nExperiment completed.")

    print("\nSaved files:")

    print(raw_path)
    print(summary_path)

    print(RESULTS / "markov_cross_entropy.png")
    print(RESULTS / "token_fertility.png")

    print(
        "\nTraining samples:",
        TRAINING_SAMPLES
    )


if __name__ == "__main__":
    main()
