"""The web server: the pages and the JSON endpoints they call.

Pages:     /upload  upload page  (app/static/upload.html)
           /photos  photos page  (app/static/photos.html)
           /review  review page  (app/static/review.html)
           /eval    eval sets page (app/static/eval.html)
           /train   datasets & models page (app/static/train.html)
           /results results page (app/static/results.html)
           /        detect page  (app/static/index.html)

Endpoints: POST /api/photos                  upload one photo: runs the pipeline (app/ingest.py)
           GET  /api/photos                  list photos in the catalog (filter by status / SKU)
           GET  /api/photos/<id>/<kind>      a photo's processed image or thumbnail
           GET  /api/classes                 the class list + accepted boxes per class
           POST /api/classes                 add a class, or change its category / prompt
           GET  /api/review                  the review queue: accepted photos + review state
           GET  /api/review/<id>             one photo's current boxes
           POST /api/review/<id>/auto-label  ask the models to propose boxes (app/review.py)
           PUT  /api/review/<id>             save the reviewer's boxes (accept / fix / reject)
           POST /api/review/<id>/reject      reject the whole photo (unusable)
           GET  /api/eval-sets               all eval sets (+ the categories you can draw from)
           POST /api/eval-sets               create a new eval set: draft v1 with a random X% of photos
           GET  /api/eval-sets/<id>          one set: its photos + candidates that could be added
           POST /api/eval-sets/<id>/<action> random / add / remove / freeze / new-version / delete (app/evalsets.py)
           GET  /api/training                what a new dataset version would contain + versions + models
           POST /api/dataset-versions        freeze and export a new dataset version (app/datasets.py)
           GET  /api/results                 models × frozen eval sets, with every run's scores
           POST /api/eval-runs               score one model on one eval set (app/evaluation.py)
           GET  /api/eval-runs/<id>          one run's per-class scores and per-photo results
           POST /api/models/<id>/use         make this the model the app uses (no restart needed)
           GET  /api/info                    which model is loaded
           POST /api/detect                  run the trained model on an uploaded photo

Start it with `uv run python -m app`.
"""

import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import catalog, datasets, evalsets, evaluation, ingest, review, storage
from app.models import detector, gdino, read_image
from app.settings import MAX_UPLOAD_MB, ROOT

STATIC_DIR = ROOT / "app" / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Runs once when the server starts: load the model the app should use.

    That's the model chosen on the Results page ("Use this model"), or else the newest one.
    """
    datasets.register_new_models()
    chosen = next((m for m in catalog.list_models() if m["promoted"]), None)
    if chosen:
        detector.load(ROOT / chosen["path"])
    else:
        detector.load_newest()
    yield


app = FastAPI(title="Object detector", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")  # css/js files


@app.middleware("http")
async def no_stale_pages(request, call_next):
    """Tell browsers to re-check files every time, so edits to app/static show up on reload."""
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response


@app.get("/upload", include_in_schema=False)
def upload_page():
    return FileResponse(STATIC_DIR / "upload.html")


@app.get("/photos", include_in_schema=False)
def photos_page():
    return FileResponse(STATIC_DIR / "photos.html")


@app.get("/review", include_in_schema=False)
def review_page():
    return FileResponse(STATIC_DIR / "review.html")


@app.get("/eval", include_in_schema=False)
def eval_page():
    return FileResponse(STATIC_DIR / "eval.html")


@app.get("/train", include_in_schema=False)
def train_page():
    return FileResponse(STATIC_DIR / "train.html")


@app.get("/results", include_in_schema=False)
def results_page():
    return FileResponse(STATIC_DIR / "results.html")


@app.get("/", include_in_schema=False)
def detect_page():
    return FileResponse(STATIC_DIR / "index.html")


def _photo_or_404(photo_id: str) -> dict:
    """Look up a photo by id, refusing anything that isn't a 64-character SHA-256 id."""
    photo = catalog.get_photo(photo_id) if re.fullmatch(r"[0-9a-f]{64}", photo_id) else None
    if photo is None:
        raise HTTPException(404, "no such photo")
    return photo


# ---------------------------------------------------------------- upload + photos (phase 1)


@app.post("/api/photos")
def upload_photo(file: UploadFile, sku: str = Form(""), uploaded_by: str = Form(""), upload_batch: str = Form("")):
    """Receive one photo and run the whole pipeline on it (app/ingest.py).

    The upload page sends photos one at a time, so a big batch shows progress photo by photo
    and no single request gets huge.
    """
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB; nothing was stored.")
    photo, already_uploaded = ingest.ingest(data, file.filename or "upload", sku.strip(),
                                            uploaded_by.strip() or "unknown", upload_batch or "unknown")
    return {"photo": photo, "already_uploaded": already_uploaded}


@app.get("/api/photos")
def list_photos(status: str = "", sku: str = ""):
    return {"photos": catalog.list_photos(status=status or None, sku=sku.strip() or None)}


@app.get("/api/photos/{photo_id}/{kind}")
def photo_image(photo_id: str, kind: str):
    """kind = "processed" (full normalized JPEG) or "thumb" (small preview)."""
    key = _photo_or_404(photo_id).get({"processed": "processed_key", "thumb": "thumb_key"}.get(kind, ""))
    if not key or not storage.exists(key):
        raise HTTPException(404, "no such image")
    return Response(storage.load(key), media_type="image/jpeg")


# ---------------------------------------------------------------- classes + review (phase 2)


@app.get("/api/classes")
def list_classes():
    return {"classes": catalog.list_classes(), "accepted_boxes": catalog.accepted_box_counts()}


class ClassIn(BaseModel):
    name: str
    category: str
    prompt: str = ""  # empty = search for the name itself


@app.post("/api/classes")
def save_class(new_class: ClassIn):
    name = new_class.name.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9 _-]*", name):
        raise HTTPException(400, "Class names use lowercase letters, digits, spaces, - and _.")
    catalog.save_class(name, new_class.category.strip().lower() or "other", new_class.prompt.strip() or name)
    return list_classes()


@app.get("/api/review")
def review_queue():
    return {"photos": catalog.review_list()}


@app.get("/api/review/{photo_id}")
def review_photo(photo_id: str):
    _photo_or_404(photo_id)
    return {"boxes": catalog.current_boxes(photo_id)}


@app.post("/api/review/{photo_id}/auto-label")
def auto_label(photo_id: str):
    if not _photo_or_404(photo_id)["processed_key"]:
        raise HTTPException(400, "This photo has no processed image.")
    used = review.auto_label(photo_id)
    if not used:
        raise HTTPException(503, "No model available to propose boxes: train a model or run scripts/download_models.py")
    return {"used": used, "boxes": catalog.current_boxes(photo_id)}


class BoxIn(BaseModel):
    """One box as the Review page sends it (corners as 0-1 fractions)."""

    id: int | None = None  # set if the box came from the catalog; empty for newly drawn boxes
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


class ReviewIn(BaseModel):
    reviewer: str
    boxes: list[BoxIn]


@app.put("/api/review/{photo_id}")
def save_review(photo_id: str, submitted: ReviewIn):
    _photo_or_404(photo_id)
    if not submitted.reviewer.strip():
        raise HTTPException(400, "Enter your name before saving.")
    try:
        counts = review.save_review(photo_id, submitted.reviewer.strip(), [b.model_dump() for b in submitted.boxes])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return counts


class RejectIn(BaseModel):
    reviewer: str


@app.post("/api/review/{photo_id}/reject")
def reject_photo(photo_id: str, body: RejectIn):
    """Mark a photo as unusable (e.g. blurry or off-topic). Its original is kept (D3)."""
    _photo_or_404(photo_id)
    catalog.update_photo(photo_id, {"status": "rejected", "check_notes": f"rejected in review by {body.reviewer.strip() or 'unknown'}"})
    return {"rejected": photo_id}


# ---------------------------------------------------------------- eval sets (phase 3)


@app.get("/api/eval-sets")
def list_eval_sets():
    categories = sorted({c["category"] for c in catalog.list_classes()})
    return {"eval_sets": catalog.list_eval_sets(), "categories": categories}


class EvalSetIn(BaseModel):
    name: str
    category: str = ""  # empty = photos of any category
    percent: float = 10
    notes: str = ""


@app.post("/api/eval-sets")
def create_eval_set(new_set: EvalSetIn):
    name = new_set.name.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9 _-]*", name):
        raise HTTPException(400, "Eval set names use lowercase letters, digits, spaces, - and _.")
    if not 0 < new_set.percent <= 100:
        raise HTTPException(400, "Percent must be between 0 and 100.")
    try:
        eval_set_id = evalsets.create(name, new_set.category.strip() or None, new_set.percent, new_set.notes.strip())
    except evalsets.EvalSetError as e:
        raise HTTPException(400, str(e))
    return evalsets.details(eval_set_id)


@app.get("/api/eval-sets/{eval_set_id}")
def eval_set_details(eval_set_id: int):
    try:
        return evalsets.details(eval_set_id)
    except evalsets.EvalSetError as e:
        raise HTTPException(404, str(e))


class EvalSetAction(BaseModel):
    count: int = 0               # for "random": how many photos to add
    photo_ids: list[str] = []    # for "add" / "remove"


@app.post("/api/eval-sets/{eval_set_id}/{action}")
def eval_set_action(eval_set_id: int, action: str, body: EvalSetAction):
    """Change an eval set. Returns the set afterwards (or, for new-version, the new draft)."""
    try:
        if action == "random":
            evalsets.add_random(eval_set_id, body.count)
        elif action == "add":
            evalsets.add_photos(eval_set_id, body.photo_ids)
        elif action == "remove":
            for photo_id in body.photo_ids:
                evalsets.remove_photo(eval_set_id, photo_id)
        elif action == "freeze":
            evalsets.freeze(eval_set_id)
        elif action == "new-version":
            eval_set_id = evalsets.new_version(eval_set_id)
        elif action == "delete":
            evalsets.delete(eval_set_id)
            return {"deleted": eval_set_id}
        else:
            raise HTTPException(404, f"unknown action {action!r}")
        return evalsets.details(eval_set_id)
    except evalsets.EvalSetError as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------- datasets + models (phase 4)


@app.get("/api/training")
def training_overview():
    """Everything the Train page shows. Also picks up newly finished training runs."""
    datasets.register_new_models()
    versions = catalog.list_dataset_versions()
    for version in versions:
        version["train_command"] = datasets.train_command(version)
    loaded = str(detector.path.relative_to(ROOT)) if detector.path else None
    return {"preview": datasets.preview(), "dataset_versions": versions, "models": catalog.list_models(),
            "loaded_model": loaded, "classes": catalog.list_classes()}


class DatasetVersionIn(BaseModel):
    val_percent: float = 15
    notes: str = ""


@app.post("/api/dataset-versions")
def freeze_dataset_version(body: DatasetVersionIn):
    if not 1 <= body.val_percent <= 50:
        raise HTTPException(400, "Validation share must be between 1 and 50%.")
    try:
        datasets.freeze(body.val_percent, body.notes.strip())
    except datasets.DatasetError as e:
        raise HTTPException(400, str(e))
    return training_overview()


# ---------------------------------------------------------------- results (phase 5)


@app.get("/api/results")
def results():
    """Everything for the results matrix: models (rows), frozen eval sets (columns), runs (cells)."""
    datasets.register_new_models()
    loaded = str(detector.path.relative_to(ROOT)) if detector.path else None
    return {"models": catalog.list_models(), "eval_sets": [e for e in catalog.list_eval_sets() if e["frozen_at"]],
            "runs": catalog.list_eval_runs(), "classes": catalog.list_classes(), "loaded_model": loaded}


class EvalRunIn(BaseModel):
    model_id: str
    eval_set_id: int


@app.post("/api/eval-runs")
def run_evaluation(body: EvalRunIn):
    """Score one model on one frozen eval set. Takes a few seconds per 100 photos."""
    try:
        evaluation.evaluate(body.model_id, body.eval_set_id)
    except evaluation.EvaluationError as e:
        raise HTTPException(400, str(e))
    return results()


@app.get("/api/eval-runs/{run_id}")
def eval_run_details(run_id: int):
    run = catalog.get_eval_run(run_id)
    if run is None:
        raise HTTPException(404, "no such eval run")
    return run


@app.post("/api/models/{model_id}/use")
def use_model(model_id: str):
    """Make this the model the app uses (Detect page + Review proposals), right away."""
    model = next((m for m in catalog.list_models() if m["id"] == model_id), None)
    if model is None:
        raise HTTPException(404, "no such model")
    detector.load(ROOT / model["path"])
    catalog.set_promoted(model_id)
    return results()


# ---------------------------------------------------------------- detect page


@app.get("/api/info")
def info():
    return {"model": detector.info(), "grounding_dino": gdino.installed()}  # model is None until you've trained one


@app.post("/api/detect")
def detect(file: UploadFile):
    try:
        image = read_image(file.file.read())
    except ValueError:
        raise HTTPException(400, "That file isn't an image I can read (JPEG, PNG or WebP).")
    try:
        detections = detector.predict(image)
    except RuntimeError as e:
        raise HTTPException(409, f"{e}. Review some photos and run training first.")
    return {"detections": detections}
