"""Where image files are kept (design.md D4).

Every image is addressed by a "key", a path-like name such as "originals/<id>.heic" or
"processed/<id>.jpg". Today keys are files under data/. In production they'll be objects in a
DigitalOcean Spaces bucket (S3-compatible): only these functions need rewriting then, nothing
else in the app touches image files directly.
"""

import os
import shutil
from pathlib import Path

from app.settings import STORAGE_DIR


def save(key: str, data: bytes) -> None:
    path = STORAGE_DIR / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def load(key: str) -> bytes:
    return (STORAGE_DIR / key).read_bytes()


def exists(key: str) -> bool:
    return (STORAGE_DIR / key).is_file()


def export_to(key: str, destination: Path) -> None:
    """Put a stored image at `destination` (a local file path) for training to read.

    Locally this makes a hard link: the same file appears in a second folder without using
    more disk space. If that's not possible (e.g. another disk), it copies. With Spaces, this
    would download the object instead.
    """
    source = STORAGE_DIR / key
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
