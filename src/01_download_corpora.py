"""
Downloads sentence files from the Leipzig Corpora Collection using its
public corpus metadata index mirrored on Hugging Face.

Important:
- This script downloads corpora locally; it does not redistribute them.
- Availability varies by language. Missing languages are logged, not silently
  replaced with a different language.
- Review the usage/license statement for each selected corpus before publication.
"""
from pathlib import Path
import csv, io, tarfile, urllib.request, urllib.error, json, re, sys
from datasets import load_dataset
from common import ROOT, CONFIG, write_json, safe_name

RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def choose_corpus(rows, lang):
    # Exact ISO-like short code first; use metadata language name as a fallback.
    code = lang["code"].lower()
    name = lang["name"].lower()
    matches = []
    for row in rows:
        short = str(row.get("language_short", "")).lower()
        lname = str(row.get("language", "")).lower()
        if short == code or lname == name:
            matches.append(row)
    if not matches:
        # Do not guess from substring matching: it can confuse language variants.
        return None

    def score(r):
        data_id = str(r.get("data_id", "")).lower()
        size = str(r.get("size", "")).upper()
        source_rank = 9
        for i, src in enumerate(CONFIG["corpus"]["preferred_sources"]):
            if src in data_id:
                source_rank = i
                break
        size_rank = 0 if size == CONFIG["corpus"]["preferred_size"] else 1
        # Prefer more recent corpus where a year can be extracted.
        years = re.findall(r"(?:19|20)\d{2}", data_id)
        year = int(years[-1]) if years else 0
        return (source_rank, size_rank, -year, data_id)
    matches.sort(key=score)
    return matches[0]

def download_and_extract(row, lang):
    data_id, url = str(row["data_id"]), str(row["url"])
    out = RAW / f"{safe_name(lang['code'])}_{safe_name(data_id)}.txt"
    if out.exists() and out.stat().st_size > 0:
        return out, data_id, url

    print(f"  Downloading {data_id}")
    req = urllib.request.Request(url, headers={"User-Agent": "academic-tokenization-study/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        archive_bytes = response.read()

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tf:
        candidates = [m for m in tf.getmembers()
                      if m.isfile() and m.name.endswith(f"{data_id}-sentences.txt")]
        if not candidates:
            candidates = [m for m in tf.getmembers()
                          if m.isfile() and m.name.endswith("-sentences.txt")]
        if not candidates:
            raise RuntimeError(f"No *-sentences.txt file found in archive {data_id}")
        member = candidates[0]
        extracted = tf.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Cannot extract {member.name}")
        raw_text = extracted.read().decode("utf-8", errors="replace")

    # Leipzig format: sentence_id<TAB>sentence
    sentences = []
    csv.field_size_limit(10_000_000)
    for row_csv in csv.reader(io.StringIO(raw_text), delimiter="\t"):
        if not row_csv:
            continue
        sentence = "\t".join(row_csv[1:]).strip() if len(row_csv) > 1 else row_csv[0].strip()
        sentence = re.sub(r"\s+", " ", sentence)
        if len(sentence) >= CONFIG["corpus"]["min_sentence_chars"] and len(sentence) <= CONFIG["corpus"]["max_sentence_chars"]:
            sentences.append(sentence)

    limit = CONFIG["corpus"]["max_sentences_per_language"]
    sentences = sentences[:limit]
    out.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    return out, data_id, url

def main():
    print("Loading Leipzig corpus metadata index...")
    ds = load_dataset(CONFIG["corpus"]["metadata_dataset"], "links", split="train")
    rows = [dict(x) for x in ds]
    manifest = {"source": CONFIG["corpus"]["metadata_dataset"], "languages": {}, "missing": []}

    for lang in CONFIG["languages"]:
        print(f"\n[{lang['name']}]")
        row = choose_corpus(rows, lang)
        if row is None:
            print("  NOT FOUND in current metadata index.")
            manifest["missing"].append(lang)
            continue
        try:
            path, data_id, url = download_and_extract(row, lang)
            manifest["languages"][lang["code"]] = {
                "name": lang["name"], "group": lang["group"],
                "data_id": data_id, "url": url,
                "file": str(path.relative_to(ROOT)),
                "sentences": sum(1 for _ in open(path, encoding="utf-8")),
                "source_language": row.get("language"),
                "year": row.get("year"), "size_label": row.get("size")
            }
            print(f"  Saved: {path.name}")
        except (urllib.error.URLError, tarfile.TarError, OSError, RuntimeError) as e:
            print(f"  ERROR: {e}")
            manifest["missing"].append({**lang, "error": str(e)})

    write_json(ROOT / "data" / "corpus_manifest.json", manifest)
    print("\nManifest:", ROOT / "data" / "corpus_manifest.json")
    print("Downloaded languages:", ", ".join(manifest["languages"].keys()) or "none")
    print("Missing languages:", ", ".join(x["code"] for x in manifest["missing"]) or "none")
    if len(manifest["languages"]) < 4:
        print("\nWARNING: Fewer than four corpora were downloaded. Check the language codes/availability.")
        print("Do not report a typological comparison until the sample is adequate.")

if __name__ == "__main__":
    main()
