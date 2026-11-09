"""Prints which requested languages actually exist in the Leipzig metadata index."""
from datasets import load_dataset
from common import CONFIG

def main():
    ds = load_dataset(CONFIG["corpus"]["metadata_dataset"], "links", split="train")
    rows = [dict(x) for x in ds]
    for lang in CONFIG["languages"]:
        matches = [r for r in rows if str(r.get("language_short","")).lower() == lang["code"].lower()
                   or str(r.get("language","")).lower() == lang["name"].lower()]
        print(f"\n{lang['name']} [{lang['code']}] — {len(matches)} corpus variants")
        for r in sorted(matches, key=lambda x: str(x.get("data_id","")))[:8]:
            print(" ", r.get("data_id"), "|", r.get("url"))

if __name__ == "__main__":
    main()
