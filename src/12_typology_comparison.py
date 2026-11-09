"""
12_typology_comparison.py

Exploratory comparison of BPE and Unigram by language and typological group.

Input:
    results/alpha_sensitivity_raw.csv

Outputs:
    results/typology_delta_raw.csv
    results/typology_delta_summary.csv
    results/typology_delta_win_counts.csv
    results/typology_delta_heatmap_alpha_0.1.png
    results/typology_delta_by_typology.png

Definition:
    delta_H = H_BPE - H_Unigram

    delta_H > 0: Unigram has lower cross-entropy.
    delta_H < 0: BPE has lower cross-entropy.

The script does not train or modify SentencePiece models.
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
INPUT_FILE = RESULTS_DIR / "alpha_sensitivity_raw.csv"

# The main alpha value used in the original experiment.
HEATMAP_ALPHA = 0.1

# Primary comparison metric. The script also calculates bits/token.
PRIMARY_METRIC = "cross_entropy_bits_per_utf8_byte"

REQUIRED_COLUMNS = {
    "code",
    "language",
    "typology_group",
    "algorithm",
    "alpha",
    "markov_order",
    "cross_entropy_bits_per_token",
    "cross_entropy_bits_per_utf8_byte",
}


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл:\n{path}\n\n"
            "Сначала запустите 11_alpha_sensitivity.py."
        )

    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "В alpha_sensitivity_raw.csv отсутствуют столбцы: "
            + ", ".join(sorted(missing))
        )

    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["markov_order"] = pd.to_numeric(
        df["markov_order"], errors="coerce"
    )
    for metric in (
        "cross_entropy_bits_per_token",
        "cross_entropy_bits_per_utf8_byte",
    ):
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df["algorithm"] = df["algorithm"].astype(str).str.lower().str.strip()
    df = df[df["algorithm"].isin(["bpe", "unigram"])].copy()

    if df.empty:
        raise ValueError("В файле нет строк для алгоритмов bpe и unigram.")

    return df


def calculate_deltas(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "code",
        "language",
        "typology_group",
        "alpha",
        "markov_order",
    ]

    # Average duplicate rows, if any, before pairing the algorithms.
    grouped = (
        df.groupby(keys + ["algorithm"], as_index=False)[
            [
                "cross_entropy_bits_per_token",
                "cross_entropy_bits_per_utf8_byte",
            ]
        ]
        .mean()
    )

    wide = grouped.pivot(
        index=keys,
        columns="algorithm",
        values=[
            "cross_entropy_bits_per_token",
            "cross_entropy_bits_per_utf8_byte",
        ],
    )

    required_pairs = [
        ("cross_entropy_bits_per_token", "bpe"),
        ("cross_entropy_bits_per_token", "unigram"),
        ("cross_entropy_bits_per_utf8_byte", "bpe"),
        ("cross_entropy_bits_per_utf8_byte", "unigram"),
    ]
    missing_pairs = [pair for pair in required_pairs if pair not in wide.columns]
    if missing_pairs:
        raise ValueError(
            "Не для всех комбинаций есть результаты обоих алгоритмов. "
            f"Отсутствуют пары: {missing_pairs}"
        )

    result = wide.reset_index()
    result.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else col
        for col in result.columns
    ]

    result = result.rename(
        columns={
            "cross_entropy_bits_per_token_bpe": "bpe_bits_per_token",
            "cross_entropy_bits_per_token_unigram": "unigram_bits_per_token",
            "cross_entropy_bits_per_utf8_byte_bpe": "bpe_bits_per_utf8_byte",
            "cross_entropy_bits_per_utf8_byte_unigram": "unigram_bits_per_utf8_byte",
        }
    )

    result["delta_bits_per_token"] = (
        result["bpe_bits_per_token"] - result["unigram_bits_per_token"]
    )
    result["delta_bits_per_utf8_byte"] = (
        result["bpe_bits_per_utf8_byte"]
        - result["unigram_bits_per_utf8_byte"]
    )

    result["better_algorithm"] = "tie"
    result.loc[result["delta_bits_per_utf8_byte"] > 0, "better_algorithm"] = "unigram"
    result.loc[result["delta_bits_per_utf8_byte"] < 0, "better_algorithm"] = "bpe"

    return result.sort_values(
        ["alpha", "typology_group", "code", "markov_order"]
    ).reset_index(drop=True)


def make_summaries(delta: pd.DataFrame):
    summary = (
        delta.groupby(["typology_group", "alpha", "markov_order"], as_index=False)
        .agg(
            languages=("code", "nunique"),
            mean_delta_bits_per_utf8_byte=("delta_bits_per_utf8_byte", "mean"),
            median_delta_bits_per_utf8_byte=("delta_bits_per_utf8_byte", "median"),
            std_delta_bits_per_utf8_byte=("delta_bits_per_utf8_byte", "std"),
            mean_delta_bits_per_token=("delta_bits_per_token", "mean"),
        )
    )

    wins = (
        delta.groupby(
            ["typology_group", "alpha", "markov_order", "better_algorithm"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("bpe", "unigram", "tie"):
        if col not in wins.columns:
            wins[col] = 0

    return summary, wins


def plot_heatmap(delta: pd.DataFrame, output_path: Path):
    subset = delta[
        (delta["alpha"].round(6) == round(HEATMAP_ALPHA, 6))
    ].copy()

    if subset.empty:
        available = sorted(delta["alpha"].dropna().unique())
        print(
            f"Предупреждение: alpha={HEATMAP_ALPHA} отсутствует. "
            f"Доступные значения: {available}. График не создан."
        )
        return

    matrix = subset.pivot(
        index="language",
        columns="markov_order",
        values="delta_bits_per_utf8_byte",
    )

    # Order languages by typology group, then alphabetically.
    order = (
        subset[["language", "typology_group"]]
        .drop_duplicates()
        .sort_values(["typology_group", "language"])["language"]
        .tolist()
    )
    matrix = matrix.reindex(order)

    fig_height = max(5.5, 0.42 * len(matrix))
    fig, ax = plt.subplots(figsize=(9, fig_height))

    max_abs = matrix.abs().max().max()
    if pd.isna(max_abs) or max_abs == 0:
        max_abs = 0.01

    image = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
    )

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([str(int(x)) for x in matrix.columns])
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Markov order")
    ax.set_ylabel("Language")
    ax.set_title(
        f"BPE − Unigram cross-entropy by language (α = {HEATMAP_ALPHA})\n"
        "Positive: Unigram lower; negative: BPE lower"
    )

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            if pd.notna(value):
                ax.text(
                    col,
                    row,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Δ cross-entropy, bits / UTF-8 byte")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_typology(summary: pd.DataFrame, output_path: Path):
    subset = summary[
        (summary["alpha"].round(6) == round(HEATMAP_ALPHA, 6))
    ].copy()

    if subset.empty:
        print("График по типологическим группам не создан: нет alpha=0.1.")
        return

    groups = sorted(subset["typology_group"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))

    for group in groups:
        part = subset[subset["typology_group"] == group].sort_values(
            "markov_order"
        )
        ax.plot(
            part["markov_order"],
            part["mean_delta_bits_per_utf8_byte"],
            marker="o",
            label=f"{group} (n={int(part['languages'].max())})",
        )

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_xticks(sorted(subset["markov_order"].unique()))
    ax.set_xlabel("Markov order")
    ax.set_ylabel("Mean Δ cross-entropy, bits / UTF-8 byte")
    ax.set_title(
        f"Mean BPE − Unigram difference by typological group (α = {HEATMAP_ALPHA})\n"
        "Positive: Unigram lower; negative: BPE lower"
    )
    ax.legend(title="Typology group", fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {INPUT_FILE}")
    df = load_data(INPUT_FILE)

    print(
        f"Loaded {len(df)} rows | "
        f"{df['code'].nunique()} languages | "
        f"alphas={sorted(df['alpha'].dropna().unique())} | "
        f"orders={sorted(df['markov_order'].dropna().unique())}"
    )

    delta = calculate_deltas(df)
    summary, wins = make_summaries(delta)

    raw_path = RESULTS_DIR / "typology_delta_raw.csv"
    summary_path = RESULTS_DIR / "typology_delta_summary.csv"
    wins_path = RESULTS_DIR / "typology_delta_win_counts.csv"
    heatmap_path = RESULTS_DIR / "typology_delta_heatmap_alpha_0.1.png"
    typology_plot_path = RESULTS_DIR / "typology_delta_by_typology.png"

    delta.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    wins.to_csv(wins_path, index=False, encoding="utf-8-sig")

    plot_heatmap(delta, heatmap_path)
    plot_typology(summary, typology_plot_path)

    print("\nAnalysis completed.")
    print(f"Raw deltas:          {raw_path}")
    print(f"Typology summary:    {summary_path}")
    print(f"Win counts:          {wins_path}")
    print(f"Language heatmap:    {heatmap_path}")
    print(f"Typology plot:       {typology_plot_path}")

    print("\nInterpretation:")
    print("  delta_H > 0: Unigram has lower cross-entropy.")
    print("  delta_H < 0: BPE has lower cross-entropy.")
    print(
        "\nCaution: typological groups have very unequal sample sizes. "
        "Treat group averages as exploratory, not as statistical proof."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
