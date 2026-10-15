"""Score a trained model on a frozen eval set (design.md §7.3, phase 5).

For every photo in the set: run the model, then compare its boxes with the frozen correct boxes.

    A predicted box (with confidence >= CONFIDENCE) is
        correct      if it has the right class and overlaps a correct box by >= OVERLAP (IoU)
        wrong class  if it overlaps a correct box well but has another class
        false alarm  otherwise
    A correct box nobody matched is missed.

Per class and overall:
    precision  = correct / all boxes the model drew        ("when it says sneaker, is it right?")
    recall     = correct / all correct boxes               ("does it find the sneakers?")
    F1         = the balance of the two (0-1, higher is better) - the headline number
    AP50       = average precision at 50% overlap, over all confidence levels (threshold-free)

Which classes are scored: those the model was trained on AND that every photo in the set was
reviewed for. Others are left out, so a class added after the eval set was frozen (and never
checked on its photos, D7) can't produce false alarms that aren't really false.
"""

import torch
from torchmetrics.detection import MeanAveragePrecision

from app import catalog, storage
from app.models import DEVICE, read_image
from app.settings import ROOT

CONFIDENCE = 0.5  # boxes below this confidence don't count as the model's answer (same default as the Detect page)
OVERLAP = 0.5     # a box counts as "the same object" if its IoU with the correct box is at least this


class EvaluationError(ValueError):
    """A request that can't be done; the message explains why."""


def evaluate(model_id: str, eval_set_id: int) -> None:
    """Run the model on every photo of the eval set and store the result as an eval run."""
    from dfine_seg import load_model

    model_row = next((m for m in catalog.list_models() if m["id"] == model_id), None)
    eval_set = catalog.get_eval_set(eval_set_id)
    if model_row is None or eval_set is None:
        raise EvaluationError("No such model or eval set.")
    if not eval_set["frozen_at"]:
        raise EvaluationError("Only frozen eval sets can be used for scoring.")

    # The model's class numbers -> catalog class ids, matched by name.
    catalog_ids = {c["name"]: c["id"] for c in catalog.list_classes()}
    model_to_catalog = {int(index): catalog_ids[name] for index, name in model_row["class_names"].items() if name in catalog_ids}

    answers = catalog.eval_set_answers(eval_set_id)
    reviewed = catalog.reviewed_classes(list(answers))
    scored = sorted(c for c in set(model_to_catalog.values()) if all(c in reviewed[p] for p in answers))
    if not scored:
        raise EvaluationError("Nothing to score: this model knows none of the classes the eval set's photos were reviewed for.")

    model = load_model(str(ROOT / model_row["path"]), conf_thresh=0.01, device=DEVICE)  # keep weak boxes too, for AP50
    # backend: the COCO scoring code that ships with D-FINE (the default, pycocotools, is not installed)
    average_precision = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[OVERLAP], class_metrics=True,
                                             backend="faster_coco_eval")
    details = []
    for photo_id, correct_boxes in answers.items():
        image = read_image(storage.load(f"processed/{photo_id}.jpg"))
        height, width = image.shape[:2]
        output = model(image, bgr=True)[0]
        predicted = [{"class_id": model_to_catalog[label], "score": score,
                      "x1": x1 / width, "y1": y1 / height, "x2": x2 / width, "y2": y2 / height}
                     for label, score, (x1, y1, x2, y2) in zip(output["labels"].tolist(), output["scores"].tolist(), output["boxes"].tolist())
                     if model_to_catalog.get(label) in scored]
        truth = [{k: b[k] for k in ("class_id", "x1", "y1", "x2", "y2")} for b in correct_boxes if b["class_id"] in scored]

        average_precision.update([_as_tensors(predicted, with_scores=True)], [_as_tensors(truth)])
        details.append({"photo_id": photo_id, **_match([p for p in predicted if p["score"] >= CONFIDENCE], truth)})

    catalog.save_eval_run(model_id, eval_set_id, scored, _metrics(details, scored, average_precision.compute()), details)


def _match(predicted: list[dict], truth: list[dict]) -> dict:
    """Pair predictions with correct boxes, most confident prediction first.

    Returns {"predicted": [... + outcome], "truth": [... + outcome]} where outcome is
    correct / wrong_class / false_alarm for predictions and found / missed for correct boxes.
    """
    found = [False] * len(truth)
    predicted = sorted(predicted, key=lambda p: -p["score"])
    for p in predicted:
        same_class = [(_iou(p, t), i) for i, t in enumerate(truth) if not found[i] and t["class_id"] == p["class_id"]]
        best_iou, best = max(same_class, default=(0.0, None))
        if best is not None and best_iou >= OVERLAP:
            found[best] = True
            p["outcome"] = "correct"
        elif any(_iou(p, t) >= OVERLAP for t in truth if t["class_id"] != p["class_id"]):
            p["outcome"] = "wrong_class"
        else:
            p["outcome"] = "false_alarm"
    for t, was_found in zip(truth, found):
        t["outcome"] = "found" if was_found else "missed"
    return {"predicted": predicted, "truth": truth}


def _metrics(details: list[dict], scored: list[int], ap_result: dict) -> dict:
    """Overall and per-class counts and scores."""
    ap_per_class = dict(zip([int(c) for c in ap_result["classes"].tolist()], ap_result["map_per_class"].tolist()))

    def summarize(class_ids: set[int]) -> dict:
        correct = sum(1 for d in details for p in d["predicted"] if p["class_id"] in class_ids and p["outcome"] == "correct")
        drawn = sum(1 for d in details for p in d["predicted"] if p["class_id"] in class_ids)
        real = sum(1 for d in details for t in d["truth"] if t["class_id"] in class_ids)
        precision = correct / drawn if drawn else 0.0
        recall = correct / real if real else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "correct": correct,
                "false_alarms": drawn - correct, "missed": real - correct, "objects": real}

    per_class = {c: {**summarize({c}), "ap50": ap_per_class.get(c, -1.0)} for c in scored}
    overall = {**summarize(set(scored)), "ap50": float(ap_result["map_50"])}
    return {"overall": overall, "per_class": per_class, "photos": len(details),
            "confidence": CONFIDENCE, "overlap": OVERLAP}


def _as_tensors(boxes: list[dict], with_scores: bool = False) -> dict:
    """Boxes in the format torchmetrics expects."""
    result = {"boxes": torch.tensor([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes], dtype=torch.float32).reshape(-1, 4),
              "labels": torch.tensor([b["class_id"] for b in boxes], dtype=torch.int64)}
    if with_scores:
        result["scores"] = torch.tensor([b["score"] for b in boxes], dtype=torch.float32)
    return result


def _iou(a: dict, b: dict) -> float:
    """Overlap of two boxes: shared area / combined area (0 = apart, 1 = identical)."""
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    shared = ix * iy
    combined = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - shared
    return shared / combined if combined > 0 else 0.0
