
import math
from collections import Counter
from pathlib import Path

import sentencepiece as spm
from importlib.util import spec_from_file_location, module_from_spec


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/eng_unigram_v8000_s13.model")
TEXT = "unhappiness"

N_SAMPLES = 20000
ALPHAS = [0.1, 1.0]


# ============================================================
# 2. LOAD EXACT LATTICE MODULE
# ============================================================

spec = spec_from_file_location(
    "pilot",
    Path("06_exact_lattice_pilot.py")
)

pilot = module_from_spec(spec)
spec.loader.exec_module(pilot)


# ============================================================
# 3. BUILD ALL THEORETICAL SEGMENTATIONS
# ============================================================

def enumerate_segmentations(sp):
    vocabulary = pilot.load_vocabulary(MODEL_PATH)
    trie = pilot.build_trie(vocabulary)

    normalized = sp.normalize(TEXT)
    edges = pilot.build_lattice(normalized, trie)

    n = len(normalized)
    paths = []

    def dfs(position, tokens, score):

        if position == n:
            paths.append((tuple(tokens), score))
            return

        for end, token, token_score in edges[position]:

            tokens.append(token)

            dfs(
                end,
                tokens,
                score + token_score
            )

            tokens.pop()

    dfs(0, [], 0.0)

    return paths


# ============================================================
# 4. THEORETICAL DISTRIBUTION FOR GIVEN ALPHA
# ============================================================

def theoretical_distribution(paths, alpha):

    log_weights = [
        alpha * score
        for _, score in paths
    ]

    log_z = pilot.logsumexp(log_weights)

    probabilities = {
        tokens: math.exp(alpha * score - log_z)
        for tokens, score in paths
    }

    return probabilities


# ============================================================
# 5. ENTROPY AND EFFECTIVE NUMBER OF SEGMENTATIONS
# ============================================================

def calculate_entropy(probabilities):

    return -math.fsum(
        p * math.log(p)
        for p in probabilities.values()
        if p > 0
    )


def effective_number(entropy):

    return math.exp(entropy)


# ============================================================
# 6. TOTAL VARIATION DISTANCE
# ============================================================

def calculate_tvd(theoretical, empirical):

    all_paths = set(theoretical) | set(empirical)

    return 0.5 * math.fsum(
        abs(
            theoretical.get(path, 0.0)
            - empirical.get(path, 0.0)
        )
        for path in all_paths
    )


# ============================================================
# 7. MONTE CARLO SAMPLING
# ============================================================

def sample_segmentations(sp, alpha):

    counts = Counter()

    for _ in range(N_SAMPLES):

        pieces = sp.sample_encode_as_pieces(
            TEXT,
            -1,
            alpha
        )

        counts[tuple(pieces)] += 1

    return counts


# ============================================================
# 8. EXPERIMENT
# ============================================================

def main():

    sp = spm.SentencePieceProcessor(
        model_file=str(MODEL_PATH)
    )

    paths = enumerate_segmentations(sp)

    print("=" * 70)
    print("EXACT SEGMENTATION DISTRIBUTION VALIDATION")
    print("=" * 70)

    print("Text:", TEXT)
    print("Model:", MODEL_PATH)
    print("Number of theoretical segmentations:", len(paths))
    print("Number of Monte Carlo samples:", N_SAMPLES)

    for alpha in ALPHAS:

        theoretical = theoretical_distribution(
            paths,
            alpha
        )

        counts = sample_segmentations(
            sp,
            alpha
        )

        empirical = {
            path: count / N_SAMPLES
            for path, count in counts.items()
        }

        # Entropy
        h_theoretical = calculate_entropy(theoretical)
        h_empirical = calculate_entropy(empirical)

        # Effective number
        neff_theoretical = effective_number(h_theoretical)
        neff_empirical = effective_number(h_empirical)

        # Normalized entropy
        n_paths = len(theoretical)

        h_norm_theoretical = (
            h_theoretical / math.log(n_paths)
            if n_paths > 1 else 0.0
        )

        h_norm_empirical = (
            h_empirical / math.log(n_paths)
            if n_paths > 1 else 0.0
        )

        # Total variation distance
        tvd = calculate_tvd(
            theoretical,
            empirical
        )

        print("\n" + "=" * 70)
        print("ALPHA:", alpha)
        print("=" * 70)

        print("\nSEGMENTATION SUPPORT")

        print("Theoretical paths:", len(theoretical))
        print("Observed paths:", len(empirical))

        print("\nENTROPY")

        print(f"Theoretical entropy: {h_theoretical:.8f}")
        print(f"Empirical entropy:   {h_empirical:.8f}")

        print("\nEFFECTIVE NUMBER OF SEGMENTATIONS")

        print(f"Theoretical N_eff: {neff_theoretical:.6f}")
        print(f"Empirical N_eff:   {neff_empirical:.6f}")

        print("\nNORMALIZED ENTROPY")

        print(f"Theoretical H_norm: {h_norm_theoretical:.6f}")
        print(f"Empirical H_norm:   {h_norm_empirical:.6f}")

        print("\nDISTRIBUTION DISTANCE")

        print(f"Total Variation Distance: {tvd:.8f}")

        print("\nTOP 10 THEORETICAL SEGMENTATIONS")

        ranked = sorted(
            theoretical.items(),
            key=lambda item: item[1],
            reverse=True
        )

        print(
            f"{'Theoretical':>13} "
            f"{'Empirical':>13} "
            f"{'Difference':>13}  Segmentation"
        )

        for tokens, theoretical_p in ranked[:10]:

            empirical_p = empirical.get(tokens, 0.0)

            print(
                f"{theoretical_p:13.6f} "
                f"{empirical_p:13.6f} "
                f"{empirical_p - theoretical_p:+13.6f}  "
                f"{tokens}"
            )


if __name__ == "__main__":
    main()
