from pathlib import Path
from collections import defaultdict
from sentencepiece import sentencepiece_model_pb2 as pb
import hashlib

MODEL_DIR = Path(__file__).resolve().parent / "models"

groups = defaultdict(list)

for path in sorted(MODEL_DIR.glob("*.model")):
    proto = pb.ModelProto()
    proto.ParseFromString(path.read_bytes())

    pieces = [
        (p.piece, round(p.score, 8), p.type)
        for p in proto.pieces
    ]

    fingerprint = hashlib.sha256(
        repr(sorted(pieces)).encode("utf-8")
    ).hexdigest()

    key = "_".join(path.stem.split("_")[:3])
    groups[key].append((path.name, fingerprint, len(pieces)))

for group, models in sorted(groups.items()):
    print(f"\n{group}")

    for name, fingerprint, size in models:
        print(name, "pieces:", size, "hash:", fingerprint[:16])

    unique = len({m[1] for m in models})
    print("Unique model fingerprints:", unique)