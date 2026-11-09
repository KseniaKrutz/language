"""
Extended analysis for the stochastic tokenization project.

Reads the corrected CSV files and existing SentencePiece Unigram models.
Does NOT train tokenizers or download corpora.

Outputs:
- metrics_extended_raw.csv: adds approximate bits/Unicode-character
- metrics_extended_summary.csv: per-language/configuration summary
- typology_descriptive.csv: descriptive typology-group averages
- sampling_boundary_metrics.csv: segmentation diversity and boundary disagreement
"""
from pathlib import Path
from collections import Counter
import json
import math
import numpy as np
import pandas as pd
import sentencepiece as spm
from common import ROOT, CONFIG

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
DATA = ROOT / "data"

RAW_IN = RESULTS / "metrics_raw_corrected.csv"
SAMPLING_IN = RESULTS / "sampling_summary_corrected.csv"

def read_sentences(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(s)
            if limit is not None and len(rows) >= limit:
                break
    return rows

def segmentation_signature(proto):
    # SentencePiece immutable-proto offsets are character/byte positions
    # in the original input; only equality and pairwise boundary comparisons
    # are used, so offsets must be consistent within a sentence.
    pieces = list(proto.pieces)
    signature = tuple((p.piece, int(p.begin), int(p.end)) for p in pieces)
    boundaries = frozenset(int(p.end) for p in pieces[:-1])
    return signature, boundaries

def pairwise_boundary_jaccard_distance(boundary_sets):
    distances = []
    for i in range(len(boundary_sets)):
        for j in range(i + 1, len(boundary_sets)):
            a, b = boundary_sets[i], boundary_sets[j]
            union = a | b
            # If both segmentations have no internal boundaries, they agree.
            d = 0.0 if not union else 1.0 - len(a & b) / len(union)
            distances.append(d)
    return float(np.mean(distances)) if distances else 0.0

def analyze_sampling(sp, sentences, repetitions, alpha, max_sentences=500):
    sentence_unique_ratios = []
    sentence_simpson_diversity = []
    sentence_boundary_distances = []
    total_samples = 0

    for sentence in sentences[:max_sentences]:
        signatures = []
        boundary_sets = []

        for _ in range(repetitions):
            pieces = sp.sample_encode_as_pieces(
                sentence,
                nbest_size=-1,
                alpha=alpha,
            )

            decoded = sp.decode_pieces(pieces)
            full_len = len(decoded)
            current_boundaries = set()

            for i in range(1, len(pieces)):
                prefix = sp.decode_pieces(pieces[:i])
                boundary = len(prefix)

                if 0 < boundary < full_len:
                    current_boundaries.add(boundary)

            signatures.append(tuple(pieces))
            boundary_sets.append(frozenset(current_boundaries))

        total_samples += len(signatures)

        freq = Counter(signatures)
        n_unique = len(freq)

        sentence_unique_ratios.append(
            n_unique / max(len(signatures), 1)
        )

        p = np.array(list(freq.values()), dtype=float) / max(len(signatures), 1)
        sentence_simpson_diversity.append(
            float(1.0 - np.sum(p ** 2))
        )

        sentence_boundary_distances.append(
            pairwise_boundary_jaccard_distance(boundary_sets)
        )

    return {
        "sampled_sentences": len(sentence_unique_ratios),
        "sampling_repetitions": repetitions,
        "total_segmentations": total_samples,
        "mean_unique_segmentation_ratio": (
            float(np.mean(sentence_unique_ratios))
            if sentence_unique_ratios else 0.0
        ),
        "mean_simpson_segmentation_diversity": (
            float(np.mean(sentence_simpson_diversity))
            if sentence_simpson_diversity else 0.0
        ),
        "mean_pairwise_boundary_jaccard_distance": (
            float(np.mean(sentence_boundary_distances))
            if sentence_boundary_distances else 0.0
        ),
    }

def main():
    if not RAW_IN.exists():
        raise SystemExit(
            f"Missing {RAW_IN}. Copy metrics_raw_corrected.csv into the project's results folder."
        )
    if not SAMPLING_IN.exists():
        raise SystemExit(
            f"Missing {SAMPLING_IN}. Copy sampling_summary_corrected.csv into the project's results folder."
        )

    raw = pd.read_csv(RAW_IN)
    sampling = pd.read_csv(SAMPLING_IN)

    # Recalculate normalizations from raw counts rather than relying on rounded fields.
    raw["cross_entropy_bits_per_unicode_char_approx"] = (
        raw["cross_entropy_bits_per_token"]
        * raw["n_tokens"] / raw["n_unicode_chars"].replace(0, np.nan)
    )
    raw["cross_entropy_bits_per_utf8_byte_approx"] = (
        raw["cross_entropy_bits_per_token"]
        * raw["n_tokens"] / raw["n_utf8_bytes"].replace(0, np.nan)
    )
    raw.to_csv(RESULTS / "metrics_extended_raw.csv", index=False, encoding="utf-8-sig")

    group_cols = [
        "language", "code", "typology_group", "algorithm",
        "vocab_size_requested", "vocab_size_actual", "markov_order"
    ]
    summary = raw.groupby(group_cols, dropna=False).agg(
        cross_entropy_bits_per_token=("cross_entropy_bits_per_token", "mean"),
        cross_entropy_bits_per_unicode_char_approx=("cross_entropy_bits_per_unicode_char_approx", "mean"),
        cross_entropy_bits_per_utf8_byte_approx=("cross_entropy_bits_per_utf8_byte_approx", "mean"),
        tokens_per_unicode_char=("tokens_per_unicode_char", "mean"),
        tokens_per_utf8_byte=("tokens_per_utf8_byte", "mean"),
        tokens_per_whitespace_unit=("tokens_per_whitespace_unit", "mean"),
        n_eval_tokens=("n_eval_tokens", "max"),
        n_train_tokens_ngram=("n_train_tokens_ngram", "max"),
    ).reset_index()
    summary.to_csv(RESULTS / "metrics_extended_summary.csv", index=False, encoding="utf-8-sig")

    # Descriptive only: typology classes are broad approximations, not causal groups.
    typology = summary.groupby(
        ["typology_group", "algorithm", "vocab_size_requested", "markov_order"],
        dropna=False
    ).agg(
        mean_bits_per_token=("cross_entropy_bits_per_token", "mean"),
        sd_across_languages_bits_per_token=("cross_entropy_bits_per_token", "std"),
        mean_bits_per_unicode_char=("cross_entropy_bits_per_unicode_char_approx", "mean"),
        mean_tokens_per_unicode_char=("tokens_per_unicode_char", "mean"),
        n_languages=("code", "nunique"),
    ).reset_index()
    typology.to_csv(RESULTS / "typology_descriptive.csv", index=False, encoding="utf-8-sig")

    # Sampling-boundary analysis uses the same 500-sentence limit and sampling
    # repetition/alpha settings as the earlier sampling run.
    manifest_path = DATA / "prepared_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = CONFIG["experiment"]
    max_sentences = 500
    repetitions_default = int(experiment.get("sampling_repetitions", 20))
    alpha_default = float(experiment.get("sampling_alpha", 0.2))

    boundary_rows = []
    for row in sampling.itertuples(index=False):
        code = row.code
        meta = manifest["languages"][code]
        test_file = ROOT / meta["test_file"]
        sentences = read_sentences(test_file, limit=max_sentences)

        model_name = str(row.model_file_reused)
        model_path = MODELS / model_name
        if not model_path.exists():
            raise SystemExit(f"Model not found: {model_path}")

        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        reps = int(getattr(row, "sampling_repetitions", repetitions_default))
        result = analyze_sampling(
            sp, sentences, reps, alpha_default, max_sentences=max_sentences
        )
        boundary_rows.append({
            "language": row.language,
            "code": code,
            "typology_group": row.typology_group,
            "algorithm": "unigram",
            "vocab_size_requested": int(row.vocab_size_requested),
            "vocab_size_actual": int(sp.get_piece_size()),
            "model_file": model_name,
            "sampling_alpha": alpha_default,
            **result,
        })
        print(f"Sampling boundary metrics: {code}, vocab={row.vocab_size_requested}")

    boundary = pd.DataFrame(boundary_rows)
    boundary.to_csv(
        RESULTS / "sampling_boundary_metrics.csv",
        index=False, encoding="utf-8-sig"
    )

    print("\nExtended analysis completed.")
    for name in [
        "metrics_extended_raw.csv",
        "metrics_extended_summary.csv",
        "typology_descriptive.csv",
        "sampling_boundary_metrics.csv",
    ]:
        print(RESULTS / name)

if __name__ == "__main__":
    main()
