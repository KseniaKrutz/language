
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

metrics = pd.read_csv(RESULTS / "metrics_extended_summary.csv")
sampling = pd.read_csv(RESULTS / "sampling_boundary_metrics.csv")
typology = pd.read_csv(RESULTS / "typology_descriptive.csv")

lines = []

def add(title, frame):
    lines.append("\n" + "=" * 70)
    lines.append(title)
    lines.append("=" * 70)
    lines.append(frame.to_string(index=False))

# 1. Проверка размеров файлов
lines.append("РАЗМЕРЫ ТАБЛИЦ")
lines.append(f"Extended summary: {metrics.shape}")
lines.append(f"Sampling boundary: {sampling.shape}")
lines.append(f"Typology: {typology.shape}")

# 2. Средние показатели энтропии по порядкам Маркова
entropy = (
    metrics.groupby("markov_order")
    ["cross_entropy_bits_per_token"]
    .mean()
    .reset_index()
)
add("СРЕДНЯЯ КРОСС-ЭНТРОПИЯ ПО ПОРЯДКАМ", entropy)

# 3. Энтропия с нормализацией на символ
char_entropy = (
    metrics.groupby(["algorithm", "vocab_size_requested"])
    ["cross_entropy_bits_per_unicode_char_approx"]
    .mean()
    .reset_index()
)
add("КРОСС-ЭНТРОПИЯ НА ЮНИКОД-СИМВОЛ", char_entropy)

# 4. Дробность токенизации по языкам
fertility = (
    metrics.groupby(["language", "algorithm", "vocab_size_requested"])
    ["tokens_per_unicode_char"]
    .mean()
    .reset_index()
)
add("ЧИСЛО ТОКЕНОВ НА СИМВОЛ", fertility)

# 5. Разнообразие сегментаций Unigram
diversity_cols = [
    "language",
    "vocab_size_requested",
    "mean_unique_segmentation_ratio",
    "mean_simpson_segmentation_diversity",
    "mean_pairwise_boundary_jaccard_distance",
]
add("РАЗНООБРАЗИЕ СЕГМЕНТАЦИЙ", sampling[diversity_cols])

# 6. Средние показатели разнообразия по группам
sampling_group = (
    sampling.groupby("typology_group")[
        [
            "mean_unique_segmentation_ratio",
            "mean_simpson_segmentation_diversity",
            "mean_pairwise_boundary_jaccard_distance",
        ]
    ]
    .mean()
    .reset_index()
)
add("РАЗНООБРАЗИЕ ПО ТИПОЛОГИЧЕСКИМ ГРУППАМ", sampling_group)

# 7. Типологические показатели
add("ТИПОЛОГИЧЕСКИЕ РЕЗУЛЬТАТЫ", typology)

report = "\n".join(lines)

output = RESULTS / "local_analysis_report.txt"
output.write_text(report, encoding="utf-8")

print(report)
print(f"\nОтчёт сохранён: {output}")
