"""Auto-labeling and human review (design.md §6.5, §6.6, D6).

auto_label(photo)   Ask the models to propose boxes for a photo and store them as `proposed`.
save_review(photo)  Turn what the reviewer submitted into accepted / rejected boxes.

Which model proposes what:
    classes your trained model knows (matched by name)  ->  your model (Detector)
    all other classes (new, or nothing trained yet)     ->  Grounding DINO, using each class's prompt
"""

from app import catalog, storage
from app.models import GroundingDino, detector, gdino, read_image

# Proposals from your trained model below this confidence aren't stored.
MODEL_MIN_SCORE = 0.35
# A new proposal overlapping an existing box by more than this is treated as the same object.
SAME_OBJECT_OVERLAP = 0.5


def auto_label(photo_id: str) -> list[str]:
    """Store proposed boxes for one photo. Returns which models were used, e.g. ["grounding-dino-tiny"]."""
    photo = catalog.get_photo(photo_id)
    image = read_image(storage.load(photo["processed_key"]))
    classes = {c["name"]: c for c in catalog.list_classes()}
    existing = catalog.current_boxes(photo_id)
    used = []

    # 1. Your trained model, for the classes it was trained on.
    model_names = set(detector.class_names.values())
    if model_names & set(classes):
        found = [{"class_id": classes[d["name"]]["id"], "score": d["score"], "box": d["box"]}
                 for d in detector.predict(image) if d["name"] in classes and d["score"] >= MODEL_MIN_SCORE]
        _store_new(photo_id, found, existing, made_by=f"model:{detector.path.parent.name}")
        used.append(f"model:{detector.path.parent.name}")

    # 2. Grounding DINO, for every class your model doesn't know.
    unknown = {c["id"]: c["prompt"] for name, c in classes.items() if name not in model_names}
    if unknown and gdino.installed():
        _store_new(photo_id, gdino.predict(image, unknown), existing, made_by=GroundingDino.NAME)
        used.append(GroundingDino.NAME)
    return used


def _store_new(photo_id: str, found: list[dict], existing: list[dict], made_by: str) -> None:
    """Store found boxes as `proposed`, skipping any that overlap a box the photo already has."""
    fresh = []
    for f in found:
        x1, y1, x2, y2 = f["box"]
        box = {"class_id": f["class_id"], "score": f["score"], "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        if not any(_overlap(box, other) > SAME_OBJECT_OVERLAP for other in existing + fresh):
            fresh.append(box)
    catalog.add_boxes(photo_id, fresh, made_by=made_by, status="proposed")


def save_review(photo_id: str, reviewer: str, submitted: list[dict]) -> dict:
    """Apply what the reviewer submitted: the photo's complete, correct set of boxes.

    Each submitted box is {id (if it came from the catalog), class_id, x1, y1, x2, y2}.
    Compared with the boxes the photo had:
      - same id, same class and position  -> accepted as is (keeps who made it: model or person)
      - anything new or changed           -> added as a new box made by the reviewer
      - old boxes not submitted unchanged -> rejected (deleted, or replaced by a corrected box)
    Returns counts of each, for the page to show.
    """
    valid_classes = {c["id"] for c in catalog.list_classes()}
    existing = {b["id"]: b for b in catalog.current_boxes(photo_id)}
    accept_ids, new_boxes = [], []

    for box in submitted:
        if box["class_id"] not in valid_classes:
            raise ValueError(f"unknown class id {box['class_id']}")
        # Keep corners inside the image and in order (left < right, top < bottom).
        x1, x2 = sorted(min(max(box[k], 0.0), 1.0) for k in ("x1", "x2"))
        y1, y2 = sorted(min(max(box[k], 0.0), 1.0) for k in ("y1", "y2"))
        if x2 - x1 < 0.001 or y2 - y1 < 0.001:
            continue  # a zero-size box, e.g. from a stray click
        clean = {"class_id": box["class_id"], "x1": x1, "y1": y1, "x2": x2, "y2": y2}

        old = existing.get(box.get("id"))
        unchanged = old is not None and old["class_id"] == clean["class_id"] and all(
            abs(old[k] - clean[k]) < 1e-6 for k in ("x1", "y1", "x2", "y2"))
        if unchanged:
            accept_ids.append(old["id"])
        else:
            new_boxes.append(clean)

    reject_ids = [box_id for box_id in existing if box_id not in accept_ids]
    # Coverage: the reviewer has now checked this photo for every class that exists.
    catalog.save_review(photo_id, reviewer, accept_ids, reject_ids, new_boxes, sorted(valid_classes))
    return {"accepted": len(accept_ids), "drawn_or_fixed": len(new_boxes), "rejected": len(reject_ids)}


def _overlap(a: dict, b: dict) -> float:
    """Shared area / combined area of two boxes (0 = apart, 1 = identical)."""
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    shared = ix * iy
    combined = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - shared
    return shared / combined if combined > 0 else 0.0
