from pathlib import Path
import json, random, re
from common import ROOT, CONFIG, set_seed, write_json

def clean_line(s):
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # Retain Unicode letters, punctuation and digits; remove control characters.
    s = "".join(ch for ch in s if ch.isprintable())
    return s

def main():
    set_seed(CONFIG["seed"])
    manifest_path = ROOT / "data" / "corpus_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Run first: python src/01_download_corpora.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_manifest = {"languages": {}}
    for code, meta in manifest["languages"].items():
        source = ROOT / meta["file"]
        lines = [clean_line(x) for x in source.read_text(encoding="utf-8").splitlines()]
        # Stable de-duplication preserves source order before seeded shuffle.
        lines = list(dict.fromkeys(x for x in lines if x))
        rng = random.Random(CONFIG["seed"])
        rng.shuffle(lines)
        split = int(len(lines) * CONFIG["experiment"]["train_fraction"])
        train, test = lines[:split], lines[split:]
        if len(test) > CONFIG["experiment"]["max_eval_sentences"]:
            test = test[:CONFIG["experiment"]["max_eval_sentences"]]
        lang_dir = ROOT / "data" / "processed" / code
        lang_dir.mkdir(parents=True, exist_ok=True)
        (lang_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
        (lang_dir / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
        out_manifest["languages"][code] = {
            **meta, "n_unique_sentences": len(lines),
            "n_train": len(train), "n_test": len(test),
            "train_file": str((lang_dir / "train.txt").relative_to(ROOT)),
            "test_file": str((lang_dir / "test.txt").relative_to(ROOT))
        }
        print(f"{code}: {len(train):,} train / {len(test):,} test")
    write_json(ROOT / "data" / "prepared_manifest.json", out_manifest)

if __name__ == "__main__":
    main()
