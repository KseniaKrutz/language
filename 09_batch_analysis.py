"""
Batch analysis of SentencePiece Unigram models.

Expected model files:
  models/{language}_unigram_v{vocab_size}_s{seed}.model

The script first checks that all 24 expected models exist.
If any model is missing, it prints the inventory and exits without
starting an incomplete experiment.

Run from the project root:
  python .\09_batch_analysis.py
"""

import csv
import math
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "results" / "batch_pilot"

LANGUAGES = ["eng", "rus", "fin", "zho"]
VOCAB_SIZES = [8000, 16000]
SEEDS = [13, 29, 47]
ALPHAS = [0.1, 1.0]

# Keep False for the first run: only check whether all 24 models exist.
# Change to True after the inventory check is complete.
RUN_ANALYSIS = True

TEST_TEXTS = {
    "eng": [
        "unhappiness",
        "natural language processing",
        "The uncertainty of tokenization",
        "internationalization",
        "machine learning models",
    ],
    "rus": [
        "неопределенность языковой модели",
        "стохастическое моделирование",
        "обработка естественного языка",
    ],
    "fin": [
        "epävarmuuden mallintaminen",
        "luonnollisen kielen käsittely",
        "koneoppimisen menetelmät",
    ],
    "zho": [
        "自然语言处理",
        "机器学习模型",
        "语言模型的不确定性",
    ],
}


# ============================================================
# 2. NUMERICAL FUNCTIONS
# ============================================================

def logsumexp(values):
    if not values:
        return float("-inf")

    maximum = max(values)
    if maximum == float("-inf"):
        return maximum

    return maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in values)
    )


def binary_entropy(p):
    p = min(1.0, max(0.0, p))
    if p == 0.0 or p == 1.0:
        return 0.0
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


# ============================================================
# 3. LOAD VOCABULARY AND BUILD LATTICE
# ============================================================

def load_vocabulary(model_path):
    proto = pb.ModelProto()
    proto.ParseFromString(model_path.read_bytes())

    return {
        piece.piece: float(piece.score)
        for piece in proto.pieces
        if piece.type == 1
    }


def build_trie(vocabulary):
    root = {}
    for token, score in vocabulary.items():
        node = root
        for char in token:
            node = node.setdefault(char, {})
        node[None] = (token, score)
    return root


def build_lattice(text, trie):
    n = len(text)
    edges = [[] for _ in range(n + 1)]

    for start in range(n):
        node = trie
        pos = start

        while pos < n and text[pos] in node:
            node = node[text[pos]]
            pos += 1

            if None in node:
                token, score = node[None]
                edges[start].append((pos, token, score))

    return edges


# ============================================================
# 4. FORWARD-BACKWARD
# ============================================================

def forward_backward(text, edges, alpha):
    n = len(text)

    forward = [float("-inf")] * (n + 1)
    forward[0] = 0.0

    for i in range(n):
        if forward[i] == float("-inf"):
            continue
        for j, token, score in edges[i]:
            forward[j] = logsumexp([
                forward[j],
                forward[i] + alpha * score,
            ])

    log_z = forward[n]
    if log_z == float("-inf"):
        raise ValueError(
            "No complete segmentation. The text may contain unknown characters."
        )

    backward = [float("-inf")] * (n + 1)
    backward[n] = 0.0

    for i in range(n - 1, -1, -1):
        candidates = [
            alpha * score + backward[j]
            for j, token, score in edges[i]
            if backward[j] != float("-inf")
        ]
        backward[i] = logsumexp(candidates)

    expected_score = 0.0
    for i in range(n):
        for j, token, score in edges[i]:
            if (
                forward[i] == float("-inf")
                or backward[j] == float("-inf")
            ):
                continue

            log_probability = (
                forward[i] + alpha * score + backward[j] - log_z
            )
            expected_score += math.exp(log_probability) * score

    entropy = log_z - alpha * expected_score
    if -1e-10 < entropy < 0.0:
        entropy = 0.0
    if entropy < 0.0:
        raise ValueError(f"Negative global entropy: {entropy}")

    boundary_probabilities = [0.0] * (n + 1)
    for i in range(1, n):
        if (
            forward[i] != float("-inf")
            and backward[i] != float("-inf")
        ):
            boundary_probabilities[i] = math.exp(
                forward[i] + backward[i] - log_z
            )

    boundary_entropies = [
        binary_entropy(q) for q in boundary_probabilities
    ]

    return {
        "log_z": log_z,
        "entropy": entropy,
        "effective_count": math.exp(entropy),
        "forward": forward,
        "backward": backward,
        "boundary_probabilities": boundary_probabilities,
        "boundary_entropies": boundary_entropies,
    }


# ============================================================
# 5. JOINT BOUNDARIES AND MUTUAL INFORMATION
# ============================================================

def log_partition_between(start, end, edges, alpha):
    if start == end:
        return 0.0

    values = [float("-inf")] * (end + 1)
    values[start] = 0.0

    for i in range(start, end):
        if values[i] == float("-inf"):
            continue

        for j, token, score in edges[i]:
            if j <= end:
                values[j] = logsumexp([
                    values[j],
                    values[i] + alpha * score,
                ])

    return values[end]


def calculate_joint_boundaries(edges, result, alpha):
    n = len(edges) - 1
    forward = result["forward"]
    backward = result["backward"]
    log_z = result["log_z"]
    q = result["boundary_probabilities"]
    joint = [[0.0] * (n + 1) for _ in range(n + 1)]
    tolerance = 1e-10

    for i in range(1, n):
        for j in range(i + 1, n):
            if forward[i] == float("-inf") or backward[j] == float("-inf"):
                continue

            middle = log_partition_between(i, j, edges, alpha)
            if middle == float("-inf"):
                continue

            p11 = math.exp(
                forward[i] + middle + backward[j] - log_z
            )

            lower = max(0.0, q[i] + q[j] - 1.0)
            upper = min(q[i], q[j])

            if p11 < lower - tolerance or p11 > upper + tolerance:
                raise ValueError(
                    f"Invalid joint probability at ({i}, {j}): {p11}"
                )

            p11 = min(upper, max(lower, p11))
            joint[i][j] = p11
            joint[j][i] = p11

    return joint


def mutual_information(q_i, q_j, p11):
    tolerance = 1e-10
    lower = max(0.0, q_i + q_j - 1.0)
    upper = min(q_i, q_j)

    if p11 < lower - tolerance or p11 > upper + tolerance:
        raise ValueError("Joint probability violates Frechet bounds.")

    p11 = min(upper, max(lower, p11))

    cells = [
        (1, 1, p11),
        (1, 0, q_i - p11),
        (0, 1, q_j - p11),
        (0, 0, 1.0 - q_i - q_j + p11),
    ]

    value = 0.0
    for b_i, b_j, p in cells:
        if p <= 0.0:
            continue

        marginal_i = q_i if b_i else 1.0 - q_i
        marginal_j = q_j if b_j else 1.0 - q_j
        denominator = marginal_i * marginal_j

        if denominator > 0.0:
            value += p * math.log(p / denominator)

    if value < -tolerance:
        raise ValueError(f"Negative mutual information: {value}")

    return max(0.0, value)


def calculate_mi_matrix(result, joint):
    q = result["boundary_probabilities"]
    n = len(q) - 1
    mi = [[0.0] * (n + 1) for _ in range(n + 1)]

    for i in range(1, n):
        for j in range(i + 1, n):
            value = mutual_information(q[i], q[j], joint[i][j])
            mi[i][j] = value
            mi[j][i] = value

    return mi


def validate_results(result, joint, mi):
    tolerance = 1e-8
    q = result["boundary_probabilities"]
    h = result["boundary_entropies"]
    n = len(q) - 1

    for i in range(1, n):
        if not (-tolerance <= q[i] <= 1.0 + tolerance):
            raise ValueError(f"Boundary probability out of range: {q[i]}")
        if h[i] < -tolerance:
            raise ValueError(f"Negative local entropy at {i}")

    sum_local = math.fsum(h[1:-1])
    if result["entropy"] > sum_local + tolerance:
        raise ValueError("Global entropy exceeds sum of local entropies.")

    for i in range(1, n):
        for j in range(i + 1, n):
            p11 = joint[i][j]
            p10 = q[i] - p11
            p01 = q[j] - p11
            p00 = 1.0 - q[i] - q[j] + p11

            if min(p00, p01, p10, p11) < -tolerance:
                raise ValueError(f"Negative joint cell at ({i}, {j})")

            if abs(p11 + p10 - q[i]) > tolerance:
                raise ValueError(f"First marginal mismatch at ({i}, {j})")
            if abs(p11 + p01 - q[j]) > tolerance:
                raise ValueError(f"Second marginal mismatch at ({i}, {j})")
            if abs(mi[i][j] - mi[j][i]) > tolerance:
                raise ValueError(f"MI symmetry failure at ({i}, {j})")
            if mi[i][j] > min(h[i], h[j]) + tolerance:
                raise ValueError(f"MI upper bound failure at ({i}, {j})")

    return sum_local


# ============================================================
# 6. MODEL INVENTORY
# ============================================================

def expected_model_paths():
    paths = []
    for language in LANGUAGES:
        for vocab_size in VOCAB_SIZES:
            for seed in SEEDS:
                filename = (
                    f"{language}_unigram_v{vocab_size}_s{seed}.model"
                )
                paths.append((language, vocab_size, seed, MODEL_DIR / filename))
    return paths


def check_model_inventory():
    expected = expected_model_paths()
    existing = []
    missing = []

    print("\nMODEL INVENTORY")
    print("-" * 72)

    for language, vocab_size, seed, path in expected:
        if path.exists():
            existing.append((language, vocab_size, seed, path))
            print(f"[OK]      {path.name}")
        else:
            missing.append(path)
            print(f"[MISSING] {path.name}")

    print("-" * 72)
    print(f"Expected: {len(expected)} | Found: {len(existing)} | Missing: {len(missing)}")

    if missing:
        print("\nBatch analysis was NOT started because some models are missing.")
        print("Train or copy the missing models into the models folder, then rerun.")
        return None

    return existing


# ============================================================
# 7. CSV OUTPUT
# ============================================================

def write_csv(path, rows):
    if not rows:
        print(f"No rows to save: {path.name}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {path} ({len(rows)} rows)")


# ============================================================
# 8. BATCH EXPERIMENT
# ============================================================

def analyze_one(model_info, model_context, text, alpha):
    language, vocab_size, seed, model_path = model_info
    sp, trie = model_context

    normalized = sp.normalize(text)
    edges = build_lattice(normalized, trie)

    result = forward_backward(normalized, edges, alpha)
    joint = calculate_joint_boundaries(edges, result, alpha)
    mi = calculate_mi_matrix(result, joint)
    sum_local = validate_results(result, joint, mi)

    sentencepiece_best = sp.encode_as_pieces(text)
    # Viterbi is not needed for the main metrics; SentencePiece's own
    # deterministic encoding is retained as a diagnostic reference.

    summary = {
        "language": language,
        "model_file": model_path.name,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": sp.get_piece_size(),
        "seed": seed,
        "alpha": alpha,
        "text": text,
        "normalized_text": normalized,
        "normalized_length": len(normalized),
        "lattice_edges": sum(len(row) for row in edges),
        "log_partition": result["log_z"],
        "global_entropy_nats": result["entropy"],
        "effective_number": result["effective_count"],
        "sum_local_entropy_nats": sum_local,
        "entropy_inequality_margin": sum_local - result["entropy"],
        "sentencepiece_segmentation": " ".join(sentencepiece_best),
        "validation_passed": True,
    }

    boundary_rows = []
    for i in range(1, len(normalized)):
        boundary_rows.append({
            "language": language,
            "model_file": model_path.name,
            "requested_vocab_size": vocab_size,
            "seed": seed,
            "alpha": alpha,
            "text": text,
            "position": i,
            "left_char": normalized[i - 1],
            "right_char": normalized[i],
            "boundary_type": (
                "word_start" if normalized[i] == "▁" else "internal"
            ),
            "boundary_probability": result["boundary_probabilities"][i],
            "boundary_entropy_nats": result["boundary_entropies"][i],
        })

    mi_rows = []
    for i in range(1, len(normalized)):
        for j in range(i + 1, len(normalized)):
            mi_rows.append({
                "language": language,
                "model_file": model_path.name,
                "requested_vocab_size": vocab_size,
                "seed": seed,
                "alpha": alpha,
                "text": text,
                "position_i": i,
                "position_j": j,
                "boundary_type_i": (
                    "word_start" if normalized[i] == "▁" else "internal"
                ),
                "boundary_type_j": (
                    "word_start" if normalized[j] == "▁" else "internal"
                ),
                "q_i": result["boundary_probabilities"][i],
                "q_j": result["boundary_probabilities"][j],
                "p_11": joint[i][j],
                "mutual_information_nats": mi[i][j],
            })

    return summary, boundary_rows, mi_rows


def main():
    models = check_model_inventory()
    if models is None:
        return

    if not RUN_ANALYSIS:
        print("\\nInventory check finished. No analysis was run.")
        print("If all 24 models are present, change RUN_ANALYSIS = True")
        print("near the top of this script and run it again.")
        return

    summary_rows = []
    boundary_rows = []
    mi_rows = []

    total_jobs = sum(len(TEST_TEXTS[lang]) for lang in LANGUAGES) * len(models) * len(ALPHAS)
    completed = 0

    print("\nSTARTING BATCH ANALYSIS")
    print(f"Models: {len(models)}")
    print(f"Alphas: {ALPHAS}")
    print(f"Total text-model-alpha combinations: {total_jobs}")

    for model_info in models:
        language = model_info[0]
        model_path = model_info[3]

        print(f"\nProcessing {model_path.name}")

        # Load SentencePiece vocabulary and build trie once per model
        vocabulary = load_vocabulary(model_path)
        trie = build_trie(vocabulary)

        # Load SentencePiece processor
        sp = spm.SentencePieceProcessor(model_file=str(model_path))

        # Store model and trie together
        model_context = (sp, trie)

        for text in TEST_TEXTS[language]:
            for alpha in ALPHAS:
                try:
                    summary, boundaries, pairs = analyze_one(
                        model_info, model_context, text, alpha
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"Failed for model={model_info[3].name}, "
                        f"text={text!r}, alpha={alpha}: {error}"
                    ) from error

                summary_rows.append(summary)
                boundary_rows.extend(boundaries)
                mi_rows.extend(pairs)

                completed += 1
                print(
                    f"  [{completed}/{total_jobs}] "
                    f"alpha={alpha} | H={summary['global_entropy_nats']:.5f} "
                    f"| N_eff={summary['effective_number']:.3f}"
                )

    write_csv(OUTPUT_DIR / "batch_summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "batch_boundary_probabilities.csv", boundary_rows)
    write_csv(OUTPUT_DIR / "batch_mutual_information_pairs.csv", mi_rows)

    print("\nBATCH ANALYSIS COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
