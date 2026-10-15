"""The two models the web app uses. This file is the only place the app talks to them.

Detector       Your trained D-FINE model (output/models/<run>/model.pt). Used by the Detect
               page, and to propose boxes on the Review page for the classes it was trained on.
GroundingDino  A ready-made model that finds things described in words. Proposes boxes on the
               Review page for classes your model doesn't know yet (e.g. before the first
               training run, or right after adding a new class). Good at finding objects,
               unreliable at telling similar types apart (sneaker vs boot), so check its types.

Both take a photo as an OpenCV array (height x width x 3, BGR, from read_image below) and
return boxes with corners as fractions of the photo size (0 to 1), which is what the pages draw.

Both run on the Mac GPU ("mps") if available. The web server answers requests on several
threads at once, so each model has a lock: only one photo goes through a model at a time.
"""

import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.settings import GDINO_DIR, ROOT, RUNS_DIR

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def read_image(data: bytes) -> np.ndarray:
    """Decode image bytes the same way D-FINE training does (OpenCV; applies EXIF rotation).

    Returns a height x width x 3 array in OpenCV's BGR channel order.
    """
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not an image OpenCV can read")
    return image


def _to_fractions(box, width: int, height: int) -> list[float]:
    """Pixel corners [x1, y1, x2, y2] -> fractions of the photo size."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1 / width, y1 / height, x2 / width, y2 / height]


class Detector:
    # D-FINE scores every guess 0-1. We return everything above this low floor and let the
    # pages decide what to show (the Detect page's slider, the Review page's 0.35 cut-off in app/review.py).
    MIN_SCORE = 0.05

    def __init__(self):
        self._lock = threading.Lock()
        self.model = None  # stays None until a model has been trained
        self.path: Path | None = None

    def load(self, path: Path) -> None:
        """Load a trained model.pt (replacing the current one, without restarting the app)."""
        from dfine_seg import load_model  # the D-FINE library; imported here because it's slow to import

        # load_model reads the weights plus the config.yaml saved next to them in the run
        # folder, which is where it gets the class names and input size from.
        model = load_model(str(path), conf_thresh=self.MIN_SCORE, device=DEVICE)
        with self._lock:
            self.model, self.path = model, path

    def load_newest(self) -> None:
        """Load the most recently trained model.pt, if there is one."""
        runs = sorted(RUNS_DIR.glob("*/model.pt"), key=lambda p: p.stat().st_mtime)
        if runs:
            self.load(runs[-1])

    @property
    def class_names(self) -> dict[int, str]:
        """Class names the loaded model was trained with, e.g. {0: "sneaker", 1: "boot"}."""
        if self.model is None:
            return {}
        return {int(class_id): name for class_id, name in self.model.names.items()}

    def info(self) -> dict | None:
        """What the pages show about the model, or None if nothing is trained yet."""
        if self.model is None:
            return None
        return {"path": str(self.path.relative_to(ROOT)), "classes": self.class_names}

    def predict(self, image: np.ndarray) -> list[dict]:
        """Run D-FINE on one photo. Returns [{cls, name, score, box: [x1, y1, x2, y2]}, ...]."""
        if self.model is None:
            raise RuntimeError("no trained shoe model yet")
        height, width = image.shape[:2]
        with self._lock:
            # Inside, D-FINE resizes the photo to 640x640, runs the network, keeps its top 300
            # guesses above MIN_SCORE, and scales the boxes back to the photo's pixel size.
            output = self.model(image, bgr=True)[0]
        names = self.class_names
        return [
            {"cls": cls, "name": names.get(cls, str(cls)), "score": score, "box": _to_fractions(box, width, height)}
            for cls, score, box in zip(output["labels"].tolist(), output["scores"].tolist(), output["boxes"].tolist())
        ]


class GroundingDino:
    MIN_SCORE = 0.3  # ignore weaker matches
    OVERLAP_LIMIT = 0.5  # if two boxes overlap more than this, keep only the higher-scoring one
    NAME = "grounding-dino-tiny"  # recorded as `made_by` on the boxes it proposes

    def __init__(self):
        self._lock = threading.Lock()
        self._processor = None  # loaded on first use: it's ~700 MB and takes a few seconds
        self._model = None

    @staticmethod
    def installed() -> bool:
        return (GDINO_DIR / "model.safetensors").exists()

    def predict(self, image: np.ndarray, prompts: dict[int, str]) -> list[dict]:
        """Find objects described by `prompts` ({class id: words}, e.g. {3: "stiletto"}).

        Returns [{class_id, score, box: [x1, y1, x2, y2]}, ...].

        All prompts go in one sentence ("sneaker . boot . stiletto ."). The model scores every
        candidate box against every word of the sentence; a box's class is the prompt whose
        words it matches best. Prompts sharing a word (e.g. "high heel shoe" and "shoe") will
        confuse it, so keep them distinct.
        """
        from torchvision.ops import nms
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        class_ids = list(prompts)
        sentence = " . ".join(prompts[c].lower().strip() for c in class_ids) + " ."
        # This model wants a Pillow image in RGB order; OpenCV gives BGR, so flip the channels.
        pil_image = Image.fromarray(image[:, :, ::-1])
        with self._lock:
            if self._model is None:
                # local_files_only: load from models/grounding-dino-tiny, never download.
                self._processor = AutoProcessor.from_pretrained(GDINO_DIR, local_files_only=True)
                self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    GDINO_DIR, local_files_only=True).to(DEVICE).eval()

            # The processor resizes the photo and turns the sentence into word-piece numbers.
            inputs = self._processor(images=pil_image, text=sentence, return_tensors="pt").to(DEVICE)
            with torch.no_grad():  # predicting only, no training: skip the bookkeeping
                outputs = self._model(**inputs)

        # Which word-piece positions belong to which prompt.
        sentence_ids = inputs.input_ids[0].tolist()
        positions = []
        for class_id in class_ids:
            pieces = self._processor.tokenizer(prompts[class_id].lower().strip(), add_special_tokens=False).input_ids
            start = next(i for i in range(len(sentence_ids)) if sentence_ids[i:i + len(pieces)] == pieces)
            positions.append(list(range(start, start + len(pieces))))

        # outputs.logits: for each of the 900 candidate boxes, a score per word-piece.
        word_scores = outputs.logits[0].sigmoid().cpu()  # 0-1, shape (900, words)
        class_scores = torch.stack([word_scores[:, p].max(dim=1).values for p in positions], dim=1)
        best_score, best_class = class_scores.max(dim=1)  # per box: its best-matching prompt
        keep = best_score >= self.MIN_SCORE

        # pred_boxes are (center x, center y, width, height) as fractions; convert to corners.
        cx, cy, w, h = outputs.pred_boxes[0].cpu()[keep].unbind(dim=1)
        corners = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1).clamp(0, 1)
        scores, classes = best_score[keep], best_class[keep]
        survivors = nms(corners, scores, self.OVERLAP_LIMIT)  # remove near-duplicate boxes
        return [{"class_id": class_ids[int(classes[i])], "score": float(scores[i]),
                 "box": [float(v) for v in corners[i]]} for i in survivors.tolist()]


# One instance of each model for the whole server.
detector = Detector()
gdino = GroundingDino()
