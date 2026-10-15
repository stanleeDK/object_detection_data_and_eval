"""Dataset versions and trained models (design.md §7.2, §7.3, phase 4).

A dataset version is a frozen snapshot of the training data:
    every photo that is accepted + reviewed + NOT in any eval set,
    with its accepted boxes at that moment,
    split at random into train (the model learns from these) and val (picks the best epoch).

Freezing writes it to data/exports/<name>/ exactly as `dfine train` wants it:

    images/<photo id>.jpg   the processed photos (hard links, so no extra disk space)
    labels/<photo id>.txt   one line per box: class x_center y_center width height (0-1);
                            an empty file = a photo with nothing in it (still useful)
    train.csv, val.csv      which image files are in which split, one name per line
    config.yaml             the project's config.yaml with this version's classes, data path
                            and run name filled in: an exact record of how to train on it

You then train in the terminal (shown on the Train page):

    DFINE_SEG_CONFIG_DIR=$PWD/data/exports/<name> uv run dfine train

($PWD = the folder you're in; D-FINE needs the config folder as a full path.)

and the result appears in output/models/<name>_<date>/, where register_new_models() finds it.
"""

import hashlib
import json
import random
from datetime import datetime

import yaml

from app import catalog, storage
from app.settings import CONFIG_PATH, EXPORTS_DIR, ROOT, RUNS_DIR


class DatasetError(ValueError):
    """A request that can't be done; the message explains why."""


def preview() -> dict:
    """What a new dataset version would contain right now (shown before freezing)."""
    pool = catalog.training_pool()
    boxes = catalog.accepted_boxes_for(pool["photo_ids"])
    return {"photos": len(pool["photo_ids"]), "boxes_per_class": _boxes_per_class(boxes),
            "empty_photos": sum(1 for b in boxes.values() if not b),
            "left_out_in_eval_sets": pool["in_eval_sets"], "left_out_not_reviewed": pool["not_reviewed"]}


def freeze(val_percent: float, notes: str) -> dict:
    """Freeze and export a new dataset version. Returns its catalog row."""
    photo_ids = catalog.training_pool()["photo_ids"]
    boxes = catalog.accepted_boxes_for(photo_ids)
    used_class_ids = sorted({box["class_id"] for photo_boxes in boxes.values() for box in photo_boxes})
    if len(photo_ids) < 2 or not used_class_ids:
        raise DatasetError("Not enough training data: review some photos with boxes first (at least 2 photos).")

    # Refuse an identical copy of the latest version: same photos with the same boxes.
    fingerprint = _fingerprint(boxes)
    versions = catalog.list_dataset_versions()  # newest first
    if versions and versions[0]["summary"].get("fingerprint") == fingerprint:
        raise DatasetError(f"Nothing changed since {versions[0]['name']}: same photos, same boxes.")

    # D-FINE numbers classes 0, 1, 2... with no gaps. Catalog class ids are permanent and can
    # have gaps (and some classes may have no boxes yet), so each version keeps its own mapping.
    class_map = {index: class_id for index, class_id in enumerate(used_class_ids)}
    index_of = {class_id: index for index, class_id in class_map.items()}
    class_names = {c["id"]: c["name"] for c in catalog.list_classes()}

    name = f"dataset-v{len(versions) + 1}"
    export = EXPORTS_DIR / name
    if export.exists():
        raise DatasetError(f"{export.relative_to(ROOT)} already exists; remove it or rename it first.")
    (export / "images").mkdir(parents=True)
    (export / "labels").mkdir()

    # Images + label files.
    for photo_id in photo_ids:
        storage.export_to(f"processed/{photo_id}.jpg", export / "images" / f"{photo_id}.jpg")
        lines = [f"{index_of[b['class_id']]} {(b['x1'] + b['x2']) / 2:.6f} {(b['y1'] + b['y2']) / 2:.6f} "
                 f"{b['x2'] - b['x1']:.6f} {b['y2'] - b['y1']:.6f}" for b in boxes[photo_id]]
        (export / "labels" / f"{photo_id}.txt").write_text("".join(line + "\n" for line in lines))

    # Random train / val split (at least one photo on each side).
    shuffled = random.sample(photo_ids, len(photo_ids))
    n_val = min(max(1, round(len(shuffled) * val_percent / 100)), len(shuffled) - 1)
    splits = {photo_id: ("val" if i < n_val else "train") for i, photo_id in enumerate(shuffled)}
    for split in ("train", "val"):
        names = [f"{photo_id}.jpg" for photo_id in photo_ids if splits[photo_id] == split]
        (export / f"{split}.csv").write_text("".join(n + "\n" for n in names))

    # config.yaml: the project's settings + this version's classes, data and run name.
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["exp_name"] = name  # the run folder becomes output/models/<name>_<date>
    config["train"]["data_path"] = f"${{train.root}}/data/exports/{name}"
    config["train"]["label_to_name"] = {index: class_names[class_id] for index, class_id in class_map.items()}
    (export / "config.yaml").write_text(
        f"# Generated by the app for {name} on {datetime.now():%Y-%m-%d %H:%M}. Train with (from the project folder):\n"
        f"#   DFINE_SEG_CONFIG_DIR=$PWD/data/exports/{name} uv run dfine train\n"
        + yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    summary = {"train": len(photo_ids) - n_val, "val": n_val, "boxes_per_class": _boxes_per_class(boxes),
               "empty_photos": sum(1 for b in boxes.values() if not b), "fingerprint": fingerprint}
    catalog.create_dataset_version(name, notes, str(export.relative_to(ROOT)), class_map, summary, splits)
    return next(v for v in catalog.list_dataset_versions() if v["name"] == name)


def train_command(version: dict, epochs: int | None = None) -> str:
    """The terminal command that trains on a dataset version (run from the project folder).

    $PWD is expanded by the shell to the current folder: D-FINE needs the config folder as a
    full path, and this keeps the command the same on any machine.
    """
    extra = f" train.epochs={epochs}" if epochs else ""
    return f"DFINE_SEG_CONFIG_DIR=$PWD/{version['export_path']} uv run dfine train{extra}"


def register_new_models() -> None:
    """Add any training run in output/models/ that the catalog doesn't know yet.

    Each run folder has the model (model.pt) and the config it was trained with (config.yaml);
    the config's data path tells which dataset version it was trained on.
    """
    known = {m["id"] for m in catalog.list_models()}
    versions = {v["export_path"]: v["id"] for v in catalog.list_dataset_versions()}
    for model_path in sorted(RUNS_DIR.glob("*/model.pt")):
        run = model_path.parent.name
        if run in known:
            continue
        config_file = model_path.parent / "config.yaml"
        train_config = (yaml.safe_load(config_file.read_text()) or {}).get("train", {}) if config_file.exists() else {}
        data_path = str(train_config.get("data_path", ""))
        version_id = next((vid for path, vid in versions.items() if data_path.endswith(path)), None)
        catalog.add_model({
            "id": run, "path": str(model_path.relative_to(ROOT)), "dataset_version_id": version_id,
            "class_names": json.dumps({str(k): v for k, v in (train_config.get("label_to_name") or {}).items()}),
            "trained_at": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(timespec="seconds"),
        })


def _boxes_per_class(boxes: dict[str, list[dict]]) -> dict[int, int]:
    """{photo id: [boxes]} -> {class id: number of boxes}."""
    counts: dict[int, int] = {}
    for photo_boxes in boxes.values():
        for box in photo_boxes:
            counts[box["class_id"]] = counts.get(box["class_id"], 0) + 1
    return counts


def _fingerprint(boxes: dict[str, list[dict]]) -> str:
    """A short code that changes whenever the photos or any of their accepted boxes change.

    Every review change creates new box ids (D15), so photo ids + box ids are enough.
    """
    text = "\n".join(f"{photo_id}:{sorted(b['id'] for b in boxes[photo_id])}" for photo_id in sorted(boxes))
    return hashlib.sha1(text.encode()).hexdigest()[:16]
