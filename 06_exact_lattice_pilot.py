
import csv
import math
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/eng_unigram_v8000_s13.model")

TEST_TEXTS = [
    "unhappiness",
    "natural language processing",
    "The uncertainty of tokenization",
    "internationalization",
    "machine learning models",
]

ALPHAS = [0.1, 1.0]

# All experiment tables will be saved here.
OUTPUT_DIR = Path("results/exact_lattice_pilot")


# ============================================================
# 2. NUMERICAL FUNCTIONS
# ============================================================

def logsumexp(values):
    """Stable calculation of log(sum(exp(values)))."""
    if not values:
        return float("-inf")

    maximum = max(values)

    if maximum == float("-inf"):
        return maximum

    return maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in values)
    )


def binary_entropy(p):
    """Entropy of a Bernoulli random variable, in nats."""
    p = min(1.0, max(0.0, p))

    if p == 0.0 or p == 1.0:
        return 0.0

    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


# ============================================================
# 3. VOCABULARY AND LATTICE
# ============================================================

def load_vocabulary(model_path):
    """Load normal pieces and their scores from SentencePiece."""
    proto = pb.ModelProto()
    proto.ParseFromString(model_path.read_bytes())

    return {
        piece.piece: float(piece.score)
        for piece in proto.pieces
        if piece.type == 1
    }


def build_trie(vocabulary):
    """Build a character trie for efficient token matching."""
    root = {}

    for token, score in vocabulary.items():
        node = root

        for char in token:
            node = node.setdefault(char, {})

        node[None] = (token, score)

    return root


def build_lattice(text, trie):
    """
    Construct a DAG segmentation lattice.

    Vertices: character positions.
    Edges: vocabulary tokens matching substrings.
    """
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
# 4. FORWARD-BACKWARD WITH ALPHA
# ============================================================

def forward_backward(text, edges, alpha=1.0):
    """
    Calculate the segmentation distribution.

    P(S | x) proportional to exp(alpha * sum(token_scores)).
    """
    n = len(text)

    # Forward pass
    forward = [float("-inf")] * (n + 1)
    forward[0] = 0.0

    for i in range(n):
        if forward[i] == float("-inf"):
            continue

        for j, token, score in edges[i]:
            candidate = forward[i] + alpha * score

            forward[j] = logsumexp([
                forward[j],
                candidate
            ])

    log_z = forward[n]

    if log_z == float("-inf"):
        raise ValueError(
            "No complete segmentation found. "
            "The string may contain unknown characters."
        )

    # Backward pass
    backward = [float("-inf")] * (n + 1)
    backward[n] = 0.0

    for i in range(n - 1, -1, -1):
        candidates = [
            alpha * score + backward[j]
            for j, token, score in edges[i]
            if backward[j] != float("-inf")
        ]

        backward[i] = logsumexp(candidates)

    # Posterior probabilities of edges
    edge_marginals = []
    expected_score = 0.0

    for i in range(n):
        for j, token, score in edges[i]:

            if (
                forward[i] == float("-inf")
                or backward[j] == float("-inf")
            ):
                continue

            log_probability = (
                forward[i]
                + alpha * score
                + backward[j]
                - log_z
            )

            probability = math.exp(log_probability)

            edge_marginals.append(
                (i, j, token, score, probability)
            )

            expected_score += probability * score

    # Global entropy
    entropy = log_z - alpha * expected_score

    if -1e-10 < entropy < 0:
        entropy = 0.0

    # Probability that a segmentation passes through each vertex.
    # Internal vertices correspond to token boundaries.
    boundary_probabilities = [0.0] * (n + 1)

    for i in range(1, n):
        if (
            forward[i] != float("-inf")
            and backward[i] != float("-inf")
        ):
            boundary_probabilities[i] = math.exp(
                forward[i] + backward[i] - log_z
            )

    # Local boundary entropy
    boundary_entropies = [
        binary_entropy(q)
        for q in boundary_probabilities
    ]

    return {
        "log_z": log_z,
        "entropy": entropy,
        "effective_count": math.exp(entropy),
        "edge_marginals": edge_marginals,
        "forward": forward,
        "backward": backward,
        "boundary_probabilities": boundary_probabilities,
        "boundary_entropies": boundary_entropies,
    }


# ============================================================
# 5. PARTITION FUNCTION BETWEEN TWO POSITIONS
# ============================================================

def log_partition_between(start, end, edges, alpha):
    """
    Calculate the log-sum of weights of all paths from start to end.

    The empty path has weight 1 when start == end.
    """
    if start == end:
        return 0.0

    values = [float("-inf")] * (end + 1)
    values[start] = 0.0

    for i in range(start, end):

        if values[i] == float("-inf"):
            continue

        for j, token, score in edges[i]:

            if j > end:
                continue

            candidate = values[i] + alpha * score

            values[j] = logsumexp([
                values[j],
                candidate
            ])

    return values[end]


# ============================================================
# 6. JOINT BOUNDARY PROBABILITIES
# ============================================================

def calculate_joint_boundaries(edges, result, alpha):
    """
    Calculate P(B_i = 1, B_j = 1 | x) for all pairs i < j.
    Only tiny floating-point violations are corrected.
    """
    n = len(edges) - 1
    forward = result["forward"]
    backward = result["backward"]
    log_z = result["log_z"]

    joint = [[0.0] * (n + 1) for _ in range(n + 1)]
    q = result["boundary_probabilities"]
    tolerance = 1e-10

    for i in range(1, n):
        for j in range(i + 1, n):
            if forward[i] == float("-inf") or backward[j] == float("-inf"):
                continue

            middle = log_partition_between(i, j, edges, alpha)
            if middle == float("-inf"):
                continue

            log_probability = forward[i] + middle + backward[j] - log_z
            p_11 = math.exp(log_probability)

            # Frechet bounds for a joint Bernoulli probability:
            # max(0, q_i + q_j - 1) <= p_11 <= min(q_i, q_j).
            lower = max(0.0, q[i] + q[j] - 1.0)
            upper = min(q[i], q[j])

            if p_11 < lower - tolerance or p_11 > upper + tolerance:
                raise ValueError(
                    f"Invalid joint probability for positions ({i}, {j}): "
                    f"p11={p_11}, allowed=[{lower}, {upper}]"
                )

            # Correct only negligible numerical deviations.
            p_11 = min(upper, max(lower, p_11))
            joint[i][j] = p_11
            joint[j][i] = p_11

    return joint


# ============================================================
# 7. MUTUAL INFORMATION BETWEEN BOUNDARIES
# ============================================================

def calculate_mutual_information(q_i, q_j, p_11):
    """Mutual information I(B_i; B_j), in nats, with validation."""
    tolerance = 1e-10

    if not (-tolerance <= q_i <= 1.0 + tolerance):
        raise ValueError(f"Invalid q_i: {q_i}")
    if not (-tolerance <= q_j <= 1.0 + tolerance):
        raise ValueError(f"Invalid q_j: {q_j}")

    q_i = min(1.0, max(0.0, q_i))
    q_j = min(1.0, max(0.0, q_j))

    lower = max(0.0, q_i + q_j - 1.0)
    upper = min(q_i, q_j)

    if p_11 < lower - tolerance or p_11 > upper + tolerance:
        raise ValueError(
            f"Invalid p11={p_11}; expected a value in [{lower}, {upper}]"
        )

    p_11 = min(upper, max(lower, p_11))

    probabilities = [
        (1, 1, p_11),
        (1, 0, q_i - p_11),
        (0, 1, q_j - p_11),
        (0, 0, 1.0 - q_i - q_j + p_11),
    ]

    result = 0.0
    for b_i, b_j, probability in probabilities:
        if probability <= 0.0:
            continue

        marginal_i = q_i if b_i == 1 else 1.0 - q_i
        marginal_j = q_j if b_j == 1 else 1.0 - q_j
        denominator = marginal_i * marginal_j

        if denominator > 0.0:
            result += probability * math.log(probability / denominator)

    if result < -tolerance:
        raise ValueError(f"Mutual information became negative: {result}")

    return max(0.0, result)


def calculate_mi_matrix(result, joint):
    """Calculate pairwise mutual information matrix."""
    q = result["boundary_probabilities"]
    n = len(q) - 1

    mi = [
        [0.0] * (n + 1)
        for _ in range(n + 1)
    ]

    for i in range(1, n):

        for j in range(i + 1, n):

            value = calculate_mutual_information(
                q[i],
                q[j],
                joint[i][j]
            )

            mi[i][j] = value
            mi[j][i] = value

    return mi


def validate_results(result, joint, mi):
    """Check probability, entropy, and mutual-information invariants."""
    tolerance = 1e-8
    q = result["boundary_probabilities"]
    h = result["boundary_entropies"]
    n = len(q) - 1

    # Boundary probabilities and entropies must be valid.
    for i in range(1, n):
        if not (-tolerance <= q[i] <= 1.0 + tolerance):
            raise ValueError(f"Boundary probability out of range at {i}: {q[i]}")
        if h[i] < -tolerance:
            raise ValueError(f"Negative local entropy at position {i}: {h[i]}")

    sum_local_entropy = math.fsum(h[1:-1])
    if result["entropy"] > sum_local_entropy + tolerance:
        raise ValueError(
            "Entropy inequality violated: global entropy exceeds "
            "the sum of local boundary entropies."
        )

    # Check joint probabilities, marginals, MI symmetry, and MI upper bound.
    for i in range(1, n):
        for j in range(i + 1, n):
            p11 = joint[i][j]
            p10 = q[i] - p11
            p01 = q[j] - p11
            p00 = 1.0 - q[i] - q[j] + p11

            if min(p00, p01, p10, p11) < -tolerance:
                raise ValueError(f"Negative joint cell probability at ({i}, {j})")

            if abs((p11 + p10) - q[i]) > tolerance:
                raise ValueError(f"Row marginal mismatch at ({i}, {j})")
            if abs((p11 + p01) - q[j]) > tolerance:
                raise ValueError(f"Column marginal mismatch at ({i}, {j})")

            if abs(mi[i][j] - mi[j][i]) > tolerance:
                raise ValueError(f"MI matrix is not symmetric at ({i}, {j})")

            if mi[i][j] > min(h[i], h[j]) + tolerance:
                raise ValueError(f"MI upper bound violated at ({i}, {j})")

    return {
        "sum_local_entropy": sum_local_entropy,
        "entropy_inequality_margin": sum_local_entropy - result["entropy"],
        "checks_passed": True,
    }


# ============================================================
# 8. VITERBI
# ============================================================

def viterbi(edges, n):
    """Find the highest-scoring segmentation."""
    best = [float("-inf")] * (n + 1)
    previous = [None] * (n + 1)

    best[0] = 0.0

    for i in range(n):

        if best[i] == float("-inf"):
            continue

        for j, token, score in edges[i]:

            candidate = best[i] + score

            if candidate > best[j]:
                best[j] = candidate
                previous[j] = (i, token)

    if previous[n] is None:
        raise ValueError("No Viterbi segmentation found.")

    result = []
    pos = n

    while pos > 0:

        item = previous[pos]

        if item is None:
            raise ValueError("Broken Viterbi path.")

        start, token = item
        result.append(token)
        pos = start

    return list(reversed(result)), best[n]


# ============================================================
# 9. ANALYSIS
# ============================================================

def analyze_text(text, sp, trie, alpha):
    normalized = sp.normalize(text)
    edges = build_lattice(normalized, trie)

    result = forward_backward(normalized, edges, alpha)
    joint = calculate_joint_boundaries(edges, result, alpha)
    mi = calculate_mi_matrix(result, joint)
    checks = validate_results(result, joint, mi)

    best_segmentation, best_score = viterbi(edges, len(normalized))
    sentencepiece_best = sp.encode_as_pieces(text)

    print("\\n" + "=" * 72)
    print("TEXT:", repr(text))
    print("ALPHA:", alpha)
    print("Normalized length:", len(normalized))
    print("Lattice edges:", sum(map(len, edges)))
    print(f"Global entropy: {result['entropy']:.8f} nats")
    print(f"Effective number: {result['effective_count']:.6f}")
    print(f"Sum of local entropies: {checks['sum_local_entropy']:.8f} nats")
    print("Validation checks: PASSED")

    if alpha == 1.0:
        print("Viterbi matches SentencePiece:",
              best_segmentation == sentencepiece_best)

    # One row per internal character position.
    boundary_rows = []
    for i in range(1, len(normalized)):
        # SentencePiece uses ▁ to mark the beginning of a word.
        boundary_type = (
            "word_start" if normalized[i] == "▁" else "internal"
        )
        boundary_rows.append({
            "text": text,
            "normalized_text": normalized,
            "alpha": alpha,
            "position": i,
            "left_char": normalized[i - 1],
            "right_char": normalized[i],
            "boundary_type": boundary_type,
            "boundary_probability": result["boundary_probabilities"][i],
            "boundary_entropy_nats": result["boundary_entropies"][i],
        })

    # One row per pair of internal boundaries.
    mi_rows = []
    for i in range(1, len(normalized)):
        for j in range(i + 1, len(normalized)):
            mi_rows.append({
                "text": text,
                "normalized_text": normalized,
                "alpha": alpha,
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

    summary_row = {
        "text": text,
        "normalized_text": normalized,
        "alpha": alpha,
        "normalized_length": len(normalized),
        "lattice_edges": sum(map(len, edges)),
        "log_partition": result["log_z"],
        "global_entropy_nats": result["entropy"],
        "effective_number": result["effective_count"],
        "sum_local_entropy_nats": checks["sum_local_entropy"],
        "entropy_inequality_margin": checks["entropy_inequality_margin"],
        "viterbi_matches_sentencepiece": (
            best_segmentation == sentencepiece_best
        ),
        "validation_passed": checks["checks_passed"],
    }

    # Keep only the five strongest dependencies in the console.
    strongest = sorted(
        mi_rows,
        key=lambda row: row["mutual_information_nats"],
        reverse=True
    )[:5]

    print("Strongest boundary dependencies:")
    for row in strongest:
        print(
            f"  ({row['position_i']}, {row['position_j']}): "
            f"{row['mutual_information_nats']:.6f} nats"
        )

    return summary_row, boundary_rows, mi_rows


# ============================================================
# 10. MAIN
# ============================================================

def write_csv(path, rows):
    """Write a list of dictionaries to CSV."""
    if not rows:
        print(f"No rows to write: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path} ({len(rows)} rows)")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading model:", MODEL_PATH)

    sp = spm.SentencePieceProcessor(model_file=str(MODEL_PATH))
    vocabulary = load_vocabulary(MODEL_PATH)
    trie = build_trie(vocabulary)

    print("Vocabulary size:", len(vocabulary))

    summary_rows = []
    boundary_rows = []
    mi_rows = []

    for text in TEST_TEXTS:
        for alpha in ALPHAS:
            summary, boundaries, mi_pairs = analyze_text(
                text, sp, trie, alpha
            )
            summary_rows.append(summary)
            boundary_rows.extend(boundaries)
            mi_rows.extend(mi_pairs)

    write_csv(OUTPUT_DIR / "summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "boundary_probabilities.csv", boundary_rows)
    write_csv(OUTPUT_DIR / "mutual_information_pairs.csv", mi_rows)

    print("\\nAll calculations completed.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
