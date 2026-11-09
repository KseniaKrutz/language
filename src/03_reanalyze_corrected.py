"""Corrected re-analysis for the stochastic tokenization experiment.

- Reuses cached SentencePiece models; does not pretend that filename seed labels
  are independent/reproducible SentencePiece random seeds.
- Uses the actual SentencePiece vocabulary size as the prediction alphabet.
- Evaluates interpolated Witten-Bell n-gram probabilities (orders 0..K).
- Saves Unigram sampling diagnostics in a separate CSV as well as JSON files.

Run from the project root:
    python src/03_reanalyze_corrected.py

Place this file in the project's src/ directory. No corpus download or data
preparation is required if prepared_manifest.json and cached models exist.
"""
from pathlib import Path
from collections import Counter, defaultdict
import json
import math

import numpy as np
import pandas as pd
import sentencepiece as spm
import matplotlib.pyplot as plt

from common import ROOT, CONFIG, write_json

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def read_sentences(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()
            if sentence:
                rows.append(sentence)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def encode_sentences(sp, sentences):
    return [sp.encode(sentence, out_type=int) for sentence in sentences]


def token_metrics(sp, sentences):
    n_tokens = sum(len(sp.encode(s, out_type=int)) for s in sentences)
    n_chars = sum(len(s) for s in sentences)
    n_bytes = sum(len(s.encode("utf-8")) for s in sentences)
    n_whitespace = sum(len(s.split()) for s in sentences)
    return {
        "n_tokens": n_tokens,
        "n_unicode_chars": n_chars,
        "n_utf8_bytes": n_bytes,
        "n_whitespace_units": n_whitespace,
        "tokens_per_unicode_char": n_tokens / max(n_chars, 1),
        "tokens_per_utf8_byte": n_tokens / max(n_bytes, 1),
        "tokens_per_whitespace_unit": n_tokens / max(n_whitespace, 1),
    }


def fit_ngram_counts(sequences, max_order, max_tokens):
    """Count n-grams without crossing sentence boundaries."""
    ngram_counts = [Counter() for _ in range(max_order + 1)]
    context_counts = [Counter() for _ in range(max_order + 1)]
    continuation_types = [defaultdict(int) for _ in range(max_order + 1)]
    seen_tokens = 0

    for sequence in sequences:
        for i, token in enumerate(sequence):
            if seen_tokens >= max_tokens:
                break
            ngram_counts[0][(token,)] += 1
            for order in range(1, max_order + 1):
                if i >= order:
                    context = tuple(sequence[i - order:i])
                    key = (context, token)
                    if ngram_counts[order][key] == 0:
                        continuation_types[order][context] += 1
                    ngram_counts[order][key] += 1
                    context_counts[order][context] += 1
            seen_tokens += 1
        if seen_tokens >= max_tokens:
            break

    return ngram_counts, context_counts, continuation_types, seen_tokens


def witten_bell_probability(token, context, order, ngram_counts,
                            context_counts, continuation_types,
                            unigram_total, vocab_size, alpha):
    """Interpolated Witten-Bell probability with add-alpha unigram base."""
    if order <= 0 or not context:
        count = ngram_counts[0].get((token,), 0)
        return (count + alpha) / (unigram_total + alpha * vocab_size)

    context = tuple(context[-order:])
    total = context_counts[order].get(context, 0)
    types = continuation_types[order].get(context, 0)

    # Unseen contexts back off to the shorter suffix context.
    if total == 0 or types == 0:
        return witten_bell_probability(
            token, context[1:], order - 1, ngram_counts,
            context_counts, continuation_types, unigram_total,
            vocab_size, alpha
        )

    count = ngram_counts[order].get((context, token), 0)
    lower_probability = witten_bell_probability(
        token, context[1:], order - 1, ngram_counts,
        context_counts, continuation_types, unigram_total,
        vocab_size, alpha
    )
    denominator = total + types
    return count / denominator + (types / denominator) * lower_probability


def evaluate_cross_entropy(test_sequences, ngram_counts, context_counts,
                           continuation_types, vocab_size, max_order,
                           alpha, max_eval_tokens):
    results = []
    unigram_total = sum(ngram_counts[0].values())

    for order in range(max_order + 1):
        total_nll = 0.0
        n_eval = 0
        for sequence in test_sequences:
            for i, token in enumerate(sequence):
                if n_eval >= max_eval_tokens:
                    break
                context = tuple(sequence[max(0, i - order):i]) if order else ()
                probability = witten_bell_probability(
                    token, context, min(order, len(context)), ngram_counts,
                    context_counts, continuation_types, unigram_total,
                    vocab_size, alpha
                )
                # Defensive clipping only protects log2 from numerical underflow.
                probability = max(probability, 1e-300)
                total_nll -= math.log2(probability)
                n_eval += 1
            if n_eval >= max_eval_tokens:
                break

        results.append({
            "markov_order": order,
            "cross_entropy_bits_per_token": total_nll / max(n_eval, 1),
            "n_eval_tokens": n_eval,
        })
    return results


def locate_cached_model(code, algorithm, vocab_size):
    """Prefer the first old cached model; seed suffix is not treated as a true seed."""
    preferred = MODELS / f"{code}_{algorithm}_v{vocab_size}_s13.model"
    if preferred.exists():
        return preferred
    candidates = sorted(MODELS.glob(f"{code}_{algorithm}_v{vocab_size}_s*.model"))
    if candidates:
        return candidates[0]
    single = MODELS / f"{code}_{algorithm}_v{vocab_size}_single.model"
    if single.exists():
        return single
    return None


def train_one_model(input_file, code, algorithm, vocab_size):
    """Fallback only: train one model if no cached model exists."""
    prefix = MODELS / f"{code}_{algorithm}_v{vocab_size}_single"
    cjk = code in {"zho", "jpn"}
    exp = CONFIG["experiment"]
    coverage = (exp["sentencepiece_character_coverage_cjk"] if cjk
                else exp["sentencepiece_character_coverage_latin"])
    sentence_count = sum(1 for _ in open(input_file, encoding="utf-8"))
    spm.SentencePieceTrainer.train(
        input=str(input_file), model_prefix=str(prefix),
        model_type=algorithm, vocab_size=vocab_size,
        character_coverage=coverage,
        input_sentence_size=min(100000, sentence_count),
        shuffle_input_sentence=True, num_threads=1,
        hard_vocab_limit=False, pad_id=0, unk_id=1,
        bos_id=-1, eos_id=-1, max_sentence_length=1000,
    )
    return Path(str(prefix) + ".model")


def sample_variability(sp, sentences, repetitions, alpha, max_sentences=500):
    per_sentence_means = []
    per_sentence_sds = []
    all_token_counts = []

    for sentence in sentences[:max_sentences]:
        lengths = []
        for _ in range(repetitions):
            ids = sp.encode(
                sentence, out_type=int, enable_sampling=True,
                nbest_size=-1, alpha=alpha
            )
            lengths.append(len(ids))
        per_sentence_means.append(float(np.mean(lengths)))
        per_sentence_sds.append(float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0)
        all_token_counts.extend(lengths)

    return {
        "sampled_sentences": len(per_sentence_means),
        "sampling_repetitions": repetitions,
        "mean_tokens_per_sentence": float(np.mean(per_sentence_means)) if per_sentence_means else None,
        "mean_within_sentence_token_count_sd": float(np.mean(per_sentence_sds)) if per_sentence_sds else None,
        "overall_sampled_token_count_sd": float(np.std(all_token_counts, ddof=1)) if len(all_token_counts) > 1 else 0.0,
    }


def main():
    manifest_path = ROOT / "data" / "prepared_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Не найден data/prepared_manifest.json. Сначала выполните 02_prepare_data.py")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exp = CONFIG["experiment"]
    max_order = int(exp["max_markov_order"])
    max_train_tokens = int(exp["max_ngram_train_tokens"])
    max_eval_tokens = int(exp["max_ngram_eval_tokens"])
    alpha = float(exp["smoothing_alpha"])
    eval_sentence_limit = min(int(exp["max_eval_sentences"]), 5000)

    metric_rows = []
    sampling_rows = []

    for code, meta in manifest["languages"].items():
        print(f"\n=== {meta['name']} ({code}) ===")
        train_file = ROOT / meta["train_file"]
        test_file = ROOT / meta["test_file"]
        train_sentences = read_sentences(train_file)
        test_sentences = read_sentences(test_file, eval_sentence_limit)
        if len(train_sentences) < 1000 or len(test_sentences) < 100:
            print("  Пропуск: недостаточно данных")
            continue

        eval_sentences = test_sentences
        for algorithm in exp["algorithms"]:
            for requested_vocab in exp["vocab_sizes"]:
                model_path = locate_cached_model(code, algorithm, int(requested_vocab))
                if model_path is None:
                    print(f"  Кэш не найден: обучаю один токенизатор {algorithm}/{requested_vocab}")
                    model_path = train_one_model(train_file, code, algorithm, int(requested_vocab))
                else:
                    print(f"  Использую кэш: {model_path.name}")

                sp = spm.SentencePieceProcessor(model_file=str(model_path))
                actual_vocab = int(sp.get_piece_size())
                train_subset = train_sentences[:min(len(train_sentences), 20000)]
                train_sequences = encode_sentences(sp, train_subset)
                test_sequences = encode_sentences(sp, eval_sentences)

                ngram_counts, context_counts, continuation_types, n_train_tokens = fit_ngram_counts(
                    train_sequences, max_order, max_train_tokens
                )
                tm = token_metrics(sp, eval_sentences)
                ce_rows = evaluate_cross_entropy(
                    test_sequences, ngram_counts, context_counts,
                    continuation_types, actual_vocab, max_order,
                    alpha, max_eval_tokens
                )

                for ce in ce_rows:
                    metric_rows.append({
                        "language": meta["name"],
                        "code": code,
                        "typology_group": meta["group"],
                        "algorithm": algorithm,
                        "vocab_size_requested": int(requested_vocab),
                        "vocab_size_actual": actual_vocab,
                        "model_file_reused": model_path.name,
                        "seed_replicates": 1,
                        "ngram_smoothing": "interpolated_witten_bell",
                        "smoothing_alpha_unigram": alpha,
                        "n_train_tokens_ngram": n_train_tokens,
                        **tm,
                        **ce,
                        "cross_entropy_bits_per_utf8_byte_approx":
                            ce["cross_entropy_bits_per_token"] * tm["tokens_per_utf8_byte"],
                    })

                if algorithm == "unigram":
                    sampling = sample_variability(
                        sp, eval_sentences, int(exp["sampling_repetitions"]),
                        float(exp["sampling_alpha"])
                    )
                    sampling_row = {
                        "language": meta["name"], "code": code,
                        "typology_group": meta["group"],
                        "algorithm": algorithm,
                        "vocab_size_requested": int(requested_vocab),
                        "vocab_size_actual": actual_vocab,
                        "model_file_reused": model_path.name,
                        **sampling,
                    }
                    sampling_rows.append(sampling_row)
                    write_json(RESULTS / f"{code}_{algorithm}_v{requested_vocab}_sampling_corrected.json", sampling_row)

    if not metric_rows:
        raise SystemExit("Результаты не получены. Проверьте manifest и каталоги данных/моделей.")

    raw = pd.DataFrame(metric_rows)
    raw.to_csv(RESULTS / "metrics_raw_corrected.csv", index=False, encoding="utf-8-sig")

    summary_cols = ["language", "code", "typology_group", "algorithm",
                    "vocab_size_requested", "vocab_size_actual", "markov_order"]
    summary = raw[summary_cols + [
        "cross_entropy_bits_per_token",
        "cross_entropy_bits_per_utf8_byte_approx",
        "tokens_per_unicode_char", "tokens_per_utf8_byte",
        "tokens_per_whitespace_unit", "n_train_tokens_ngram", "n_eval_tokens"
    ]].copy()
    summary.to_csv(RESULTS / "metrics_summary_corrected.csv", index=False, encoding="utf-8-sig")

    sampling_df = pd.DataFrame(sampling_rows)
    sampling_df.to_csv(RESULTS / "sampling_summary_corrected.csv", index=False, encoding="utf-8-sig")

    # Diagnostic plots: keep each figure readable by separating algorithms/vocab sizes.
    for algorithm in exp["algorithms"]:
        for requested_vocab in exp["vocab_sizes"]:
            subset = summary[(summary.algorithm == algorithm) &
                             (summary.vocab_size_requested == int(requested_vocab))]
            if subset.empty:
                continue
            plt.figure(figsize=(10, 6))
            for code, group in subset.groupby("code"):
                group = group.sort_values("markov_order")
                plt.plot(group.markov_order, group.cross_entropy_bits_per_utf8_byte_approx,
                         marker="o", label=code)
            plt.xlabel("Порядок контекста k")
            plt.ylabel("Приближённая кросс-энтропия (бит/UTF-8 байт)")
            plt.title(f"{algorithm.upper()}, requested vocab={requested_vocab}")
            plt.legend(fontsize=8, ncol=3)
            plt.tight_layout()
            plt.savefig(RESULTS / f"cross_entropy_utf8_{algorithm}_v{requested_vocab}_corrected.png", dpi=180)
            plt.close()

    print("\nГотово. Созданы файлы:")
    print(RESULTS / "metrics_raw_corrected.csv")
    print(RESULTS / "metrics_summary_corrected.csv")
    print(RESULTS / "sampling_summary_corrected.csv")
    print("Графики cross_entropy_utf8_*_corrected.png")
    print("\nВажно: старые модели переиспользованы; три старых seed-метки не считаются независимыми повторами.")


if __name__ == "__main__":
    main()
