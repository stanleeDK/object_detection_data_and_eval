"""Download the models this project uses into models/ (skips files already there).

    uv run python scripts/download_models.py

- models/grounding-dino-tiny/  zero-shot shoe finder for the label page (~690 MB)
- models/pretrained/           D-FINE starting weights for training (~40 MB per size)
"""

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
D_FINE_SIZES = sys.argv[1:] or ["s"]  # e.g. `download_models.py s m` for more sizes

print("Grounding DINO -> models/grounding-dino-tiny/")
snapshot_download("IDEA-Research/grounding-dino-tiny", local_dir=ROOT / "models" / "grounding-dino-tiny")

for size in D_FINE_SIZES:
    print(f"D-FINE {size} (Objects365 -> COCO) -> models/pretrained/")
    for filename in ("config.json", f"dfine_{size}_obj2coco.pt"):
        hf_hub_download("ArgoSA/D-FINE-seg", filename, local_dir=ROOT / "models" / "pretrained")
print("Done.")
