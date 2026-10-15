"""Folder locations and settings, shared by the web app and scripts/."""

from pathlib import Path

# The project folder (this file is app/settings.py, so two levels up).
ROOT = Path(__file__).resolve().parent.parent

# D-FINE's training config. Its class names (train.label_to_name) were copied into the
# catalog's classes table the first time the catalog was created; since then the catalog is
# the source of truth for classes (managed on the Review page).
CONFIG_PATH = ROOT / "config.yaml"

# Each `dfine train` run creates output/models/<run name>/model.pt. The app uses the newest.
RUNS_DIR = ROOT / "output" / "models"

# The zero-shot model that proposes boxes for classes your own model doesn't know yet.
GDINO_DIR = ROOT / "models" / "grounding-dino-tiny"

# ---- Upload pipeline (design.md §6) ----

# The catalog: a SQLite database holding facts about every photo (not the images themselves).
CATALOG_PATH = ROOT / "data" / "catalog.sqlite"

# Image files: data/originals/ (uploads exactly as received), data/processed/ (normalized
# JPEGs) and data/thumbs/ (small copies for the web pages). See app/storage.py.
STORAGE_DIR = ROOT / "data"

MAX_UPLOAD_MB = 50  # bigger uploads are refused before anything is stored
JPEG_QUALITY = 95  # quality of the normalized JPEGs (1-100; 95 = visually lossless)
THUMB_SIZE = 400  # long side (px) of the thumbnails shown on the Photos/Upload pages
MIN_SHORT_SIDE = 480  # photos whose short side is smaller than this (px) are rejected
# Two photos are near-duplicates if their 64-bit perceptual hashes differ in at most this
# many bits. 0 = pixel-identical look; ~5 catches burst shots and re-saved copies.
NEAR_DUPLICATE_BITS = 5

# ---- Training (design.md §7.2) ----

# Each frozen dataset version is written here as data/exports/<name>/ in the layout
# `dfine train` reads: images/, labels/, train.csv, val.csv and its own config.yaml.
EXPORTS_DIR = ROOT / "data" / "exports"
