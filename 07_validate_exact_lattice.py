
import math
import importlib.util
from pathlib import Path

# Import functions from the previous pilot script
spec = importlib.util.spec_from_file_location(
    "pilot",
    Path("06_exact_lattice_pilot.py")
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

MODEL_PATH = Path("models/eng_unigram_v8000_s13.model")
TEXT = "unhappiness"


def enumerate_paths(edges, n):
    """Enumerate all complete paths in a small lattice."""
    paths = []

    def dfs(position, tokens, score, used_edges):
        if position == n:
            paths.append((tokens.copy(), score, used_edges.copy()))
            return

        for end, token, token_score in edges[position]:
            tokens.append(token)
            used_edges.append((position, end, token))
            dfs(
                end,
                tokens,
                score + token_score,
                used_edges
            )
            tokens.pop()
            used_edges.pop()

    dfs(0, [], 0.0, [])
    return paths


def main():
    sp = pilot.spm.SentencePieceProcessor(
        model_file=str(MODEL_PATH)
    )

    vocabulary = pilot.load_vocabulary(MODEL_PATH)
    trie = pilot.build_trie(vocabulary)

    normalized = sp.normalize(TEXT)
    edges = pilot.build_lattice(normalized, trie)

    # Forward-Backward result
    fb = pilot.forward_backward(normalized, edges)

    # Exhaustive enumeration
    paths = enumerate_paths(edges, len(normalized))

    log_weights = [score for _, score, _ in paths]
    log_z = pilot.logsumexp(log_weights)

    probabilities = [
        math.exp(score - log_z)
        for score in log_weights
    ]

    entropy = -sum(
        p * math.log(p)
        for p in probabilities
        if p > 0
    )

    effective_count = math.exp(entropy)

    print("TEXT:", TEXT)
    print("NORMALIZED:", normalized)
    print("NUMBER OF COMPLETE PATHS:", len(paths))

    print("\nFORWARD-BACKWARD:")
    print("log Z:", fb["log_z"])
    print("Entropy:", fb["entropy"])
    print("Effective count:", fb["effective_count"])

    print("\nEXHAUSTIVE ENUMERATION:")
    print("log Z:", log_z)
    print("Entropy:", entropy)
    print("Effective count:", effective_count)

    print("\nDIFFERENCES:")
    print("log Z difference:", abs(fb["log_z"] - log_z))
    print("Entropy difference:", abs(fb["entropy"] - entropy))
    print(
        "Effective count difference:",
        abs(fb["effective_count"] - effective_count)
    )

    print("\nTOP 10 PATHS:")
    ranked = sorted(paths, key=lambda x: x[1], reverse=True)

    for i, (tokens, score, _) in enumerate(ranked[:10], 1):
        probability = math.exp(score - log_z)
        print(
            f"{i:2}. P={probability:.8f} "
            f"score={score:.6f} "
            f"{tokens}"
        )


if __name__ == "__main__":
    main()
