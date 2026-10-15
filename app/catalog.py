"""The catalog: a SQLite database of facts about every photo (design.md §12, D4).

SQLite is a whole database in one file (data/catalog.sqlite), with no server to run.

Tables so far:
    photos         one row per uploaded photo (phase 1)
    classes        the global list of object types: sneaker, boot, sofa... (phase 2)
    boxes          every box ever proposed by a model or drawn by a person, with its status (phase 2)
    photo_reviews  which classes each photo has been fully checked for ("coverage", phase 2)
    eval_sets, eval_set_photos, eval_set_boxes
                   named, versioned eval sets; the boxes are copied in when a set is frozen (phase 3)
    dataset_versions, dataset_version_photos
                   frozen training snapshots, exported for `dfine train` (phase 4)
    models         trained models found in output/models, and what they were trained on (phase 4)
    eval_runs      a model's score on one frozen eval set, with per-photo results (phase 5)

Every function opens the database, does one thing, and closes it again. That's simple and
safe when the web server handles several requests at once.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime

import yaml

from app.settings import CATALOG_PATH, CONFIG_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id             TEXT PRIMARY KEY,  -- SHA-256 of the original file's bytes (pipeline step 1)
    original_name  TEXT NOT NULL,     -- file name on the uploader's device, e.g. IMG_0042.HEIC
    original_key   TEXT NOT NULL,     -- storage key of the untouched original (app/storage.py)
    processed_key  TEXT,              -- storage key of the normalized JPEG (empty if it couldn't be decoded)
    thumb_key      TEXT,              -- storage key of the small preview for web pages
    uploaded_by    TEXT NOT NULL,     -- name typed on the upload page (accounts come in phase 6)
    upload_batch   TEXT NOT NULL,     -- one id per press of the Upload button
    uploaded_at    TEXT NOT NULL,     -- ISO date-time
    sku            TEXT,              -- optional product code, one per photo (D11)
    device         TEXT,              -- camera make + model from EXIF, if present
    taken_at       TEXT,              -- when the photo was taken, from EXIF, if present
    width          INTEGER,           -- size of the processed image (after rotation)
    height         INTEGER,
    phash          TEXT,              -- perceptual hash as 16 hex digits (pipeline step 14)
    duplicate_of   TEXT,              -- id of the photo this nearly duplicates, if dropped for that (D9)
    status         TEXT NOT NULL,     -- processing / accepted / rejected
    check_notes    TEXT               -- why it was rejected, in plain words
);

CREATE TABLE IF NOT EXISTS classes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,  -- permanent number; AUTOINCREMENT = never reused
    name      TEXT NOT NULL UNIQUE,               -- e.g. sneaker, sofa
    category  TEXT NOT NULL,                      -- e.g. footwear, furniture
    prompt    TEXT NOT NULL,                      -- words Grounding DINO searches for, e.g. "stiletto" for heel
    added_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS boxes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id     TEXT NOT NULL REFERENCES photos(id),
    class_id     INTEGER NOT NULL REFERENCES classes(id),
    x1 REAL NOT NULL, y1 REAL NOT NULL,   -- corners as fractions of the image size (0 to 1)
    x2 REAL NOT NULL, y2 REAL NOT NULL,
    made_by      TEXT NOT NULL,           -- "grounding-dino-tiny", "model:<run name>" or "human:<name>" (D6)
    score        REAL,                    -- the model's confidence; empty for boxes drawn by a person
    status       TEXT NOT NULL,           -- proposed (machine guess) / accepted (confirmed) / rejected (D6)
    created_at   TEXT NOT NULL,
    reviewed_at  TEXT                     -- when it was accepted or rejected
);

CREATE TABLE IF NOT EXISTS photo_reviews (
    photo_id     TEXT NOT NULL REFERENCES photos(id),
    class_id     INTEGER NOT NULL REFERENCES classes(id),
    reviewed_at  TEXT NOT NULL,
    reviewed_by  TEXT NOT NULL,
    PRIMARY KEY (photo_id, class_id)      -- "this photo was fully checked for this class"
);

CREATE TABLE IF NOT EXISTS eval_sets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,            -- e.g. shoes-eval
    version     INTEGER NOT NULL,         -- 1, 2, 3... (a changed set is a new version)
    category    TEXT,                     -- photos are drawn from this category (e.g. footwear); empty = any
    notes       TEXT,
    created_at  TEXT NOT NULL,
    frozen_at   TEXT,                     -- empty while it's a draft; set once frozen (then it never changes)
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS eval_set_photos (
    eval_set_id  INTEGER NOT NULL REFERENCES eval_sets(id),
    photo_id     TEXT NOT NULL REFERENCES photos(id),
    PRIMARY KEY (eval_set_id, photo_id)
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,    -- dataset-v1, dataset-v2...
    created_at   TEXT NOT NULL,
    notes        TEXT,
    export_path  TEXT NOT NULL,           -- folder written for `dfine train`, relative to the project
    class_map    TEXT NOT NULL,           -- JSON: model class number -> catalog class id, e.g. {"0": 1, "1": 3}
    summary      TEXT NOT NULL            -- JSON: photo and box counts at freeze time (shown on the page)
);

CREATE TABLE IF NOT EXISTS dataset_version_photos (
    dataset_version_id  INTEGER NOT NULL REFERENCES dataset_versions(id),
    photo_id            TEXT NOT NULL REFERENCES photos(id),
    split               TEXT NOT NULL,    -- train (learned from) / val (used to pick the best epoch)
    PRIMARY KEY (dataset_version_id, photo_id)
);

CREATE TABLE IF NOT EXISTS models (
    id                  TEXT PRIMARY KEY, -- training run folder name, e.g. dataset-v1_2026-10-15
    path                TEXT NOT NULL,    -- output/models/<run>/model.pt
    dataset_version_id  INTEGER REFERENCES dataset_versions(id),  -- empty if trained on data from outside the app
    class_names         TEXT,             -- JSON from the run's config: model class number -> name
    trained_at          TEXT,             -- when model.pt was last written
    registered_at       TEXT NOT NULL,    -- when the app first found it
    promoted            INTEGER NOT NULL DEFAULT 0  -- 1 = the model the app uses (at most one)
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id        TEXT NOT NULL REFERENCES models(id),
    eval_set_id     INTEGER NOT NULL REFERENCES eval_sets(id),
    run_at          TEXT NOT NULL,
    scored_classes  TEXT NOT NULL,  -- JSON list of catalog class ids that were scored (see app/evaluation.py)
    metrics         TEXT NOT NULL,  -- JSON: overall + per class precision / recall / F1 / AP50 / counts
    details         TEXT NOT NULL,  -- JSON: per photo, every box and whether it was found / missed / a false alarm
    UNIQUE (model_id, eval_set_id)  -- re-running replaces the old result (the same model on the same set gives the same answer)
);

CREATE TABLE IF NOT EXISTS eval_set_boxes (   -- the correct answers, copied from `boxes` at freeze time
    eval_set_id  INTEGER NOT NULL REFERENCES eval_sets(id),
    photo_id     TEXT NOT NULL REFERENCES photos(id),
    class_id     INTEGER NOT NULL REFERENCES classes(id),
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CATALOG_PATH)
    connection.row_factory = sqlite3.Row  # rows behave like dicts: row["status"]
    connection.executescript(SCHEMA)  # creates missing tables; does nothing for existing ones
    _seed_classes(connection)
    return connection


def _seed_classes(db: sqlite3.Connection) -> None:
    """The very first time, copy the class names from config.yaml into the classes table.

    After that the catalog is the source of truth for classes, managed on the Review page.
    """
    if db.execute("SELECT COUNT(*) FROM classes").fetchone()[0]:
        return
    names = yaml.safe_load(CONFIG_PATH.read_text())["train"]["label_to_name"]
    with db:
        for _, name in sorted(names.items(), key=lambda kv: int(kv[0])):
            db.execute("INSERT INTO classes (name, category, prompt, added_at) VALUES (?, ?, ?, ?)",
                       (str(name), "footwear", str(name), _now()))


# ---------------------------------------------------------------- photos


def get_photo(photo_id: str) -> dict | None:
    with closing(_connect()) as db:
        row = db.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
    return dict(row) if row else None


def add_photo(fields: dict) -> None:
    columns = ", ".join(fields)
    placeholders = ", ".join("?" for _ in fields)
    with closing(_connect()) as db, db:  # the second `db` commits the change when the block ends
        db.execute(f"INSERT INTO photos ({columns}) VALUES ({placeholders})", list(fields.values()))


def update_photo(photo_id: str, fields: dict) -> None:
    assignments = ", ".join(f"{column} = ?" for column in fields)
    with closing(_connect()) as db, db:
        db.execute(f"UPDATE photos SET {assignments} WHERE id = ?", [*fields.values(), photo_id])


def list_photos(status: str | None = None, sku: str | None = None) -> list[dict]:
    """Photos, newest first, optionally only one status and/or SKUs containing `sku`.

    Each row also gets `eval_sets`: the eval sets it's in, e.g. "shoes-eval v1, shoes-eval v2".
    """
    query, params = """SELECT p.*, (SELECT GROUP_CONCAT(e.name || ' v' || e.version, ', ')
                                     FROM eval_set_photos m JOIN eval_sets e ON e.id = m.eval_set_id
                                     WHERE m.photo_id = p.id) AS eval_sets
                       FROM photos p WHERE 1 = 1""", []
    if status:
        query += " AND status = ?"
        params.append(status)
    if sku:
        query += " AND sku LIKE ?"
        params.append(f"%{sku}%")
    query += " ORDER BY uploaded_at DESC, original_name"
    with closing(_connect()) as db:
        return [dict(row) for row in db.execute(query, params)]


def accepted_phashes() -> list[tuple[str, str]]:
    """(id, perceptual hash) of every accepted photo, for the near-duplicate check."""
    with closing(_connect()) as db:
        rows = db.execute("SELECT id, phash FROM photos WHERE status = 'accepted' AND phash IS NOT NULL")
        return [(row["id"], row["phash"]) for row in rows]


# ---------------------------------------------------------------- classes


def list_classes() -> list[dict]:
    """All classes, oldest first. The Review page's keys 1-9 follow this order."""
    with closing(_connect()) as db:
        return [dict(row) for row in db.execute("SELECT * FROM classes ORDER BY id")]


def save_class(name: str, category: str, prompt: str) -> None:
    """Add a new class, or change the category/prompt of an existing one (matched by name).

    Classes are never renamed or deleted here: their ids are permanent (design.md §5).
    """
    with closing(_connect()) as db, db:
        db.execute(
            "INSERT INTO classes (name, category, prompt, added_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET category = excluded.category, prompt = excluded.prompt",
            (name, category, prompt, _now()))


def accepted_box_counts() -> dict[int, int]:
    """class id -> number of accepted boxes, shown next to each class on the Review page."""
    with closing(_connect()) as db:
        rows = db.execute("SELECT class_id, COUNT(*) AS n FROM boxes WHERE status = 'accepted' GROUP BY class_id")
        return {row["class_id"]: row["n"] for row in rows}


# ---------------------------------------------------------------- boxes and reviews


def current_boxes(photo_id: str) -> list[dict]:
    """A photo's live boxes: proposed and accepted (rejected ones are history only)."""
    with closing(_connect()) as db:
        rows = db.execute("SELECT * FROM boxes WHERE photo_id = ? AND status IN ('proposed', 'accepted') ORDER BY id",
                          (photo_id,))
        return [dict(row) for row in rows]


def add_boxes(photo_id: str, boxes: list[dict], made_by: str, status: str) -> None:
    """Store boxes, each {class_id, x1, y1, x2, y2, score (optional)}."""
    now = _now()
    reviewed_at = now if status != "proposed" else None
    with closing(_connect()) as db, db:
        db.executemany(
            "INSERT INTO boxes (photo_id, class_id, x1, y1, x2, y2, made_by, score, status, created_at, reviewed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(photo_id, b["class_id"], b["x1"], b["y1"], b["x2"], b["y2"], made_by, b.get("score"), status, now, reviewed_at)
             for b in boxes])


def save_review(photo_id: str, reviewer: str, accept_ids: list[int], reject_ids: list[int],
                new_boxes: list[dict], class_ids: list[int]) -> None:
    """Record one review of a photo, all in one transaction (all of it is saved, or none):

    - boxes in accept_ids become `accepted` (kept exactly as the model or a person made them)
    - boxes in reject_ids become `rejected` (deleted or changed by the reviewer)
    - new_boxes are added as `accepted`, made by the reviewer (drawn or corrected by hand)
    - the photo is marked as fully checked for every class in class_ids (coverage)
    """
    now = _now()
    with closing(_connect()) as db, db:
        db.executemany("UPDATE boxes SET status = 'accepted', reviewed_at = ? WHERE id = ? AND photo_id = ?",
                       [(now, box_id, photo_id) for box_id in accept_ids])
        db.executemany("UPDATE boxes SET status = 'rejected', reviewed_at = ? WHERE id = ? AND photo_id = ?",
                       [(now, box_id, photo_id) for box_id in reject_ids])
        db.executemany(
            "INSERT INTO boxes (photo_id, class_id, x1, y1, x2, y2, made_by, score, status, created_at, reviewed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'accepted', ?, ?)",
            [(photo_id, b["class_id"], b["x1"], b["y1"], b["x2"], b["y2"], f"human:{reviewer}", now, now) for b in new_boxes])
        db.executemany(
            "INSERT INTO photo_reviews (photo_id, class_id, reviewed_at, reviewed_by) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(photo_id, class_id) DO UPDATE SET reviewed_at = excluded.reviewed_at, reviewed_by = excluded.reviewed_by",
            [(photo_id, class_id, now, reviewer) for class_id in class_ids])


def review_list() -> list[dict]:
    """Every accepted photo with its review state, oldest upload first (the review queue)."""
    with closing(_connect()) as db:
        rows = db.execute("""
            SELECT p.id, p.original_name, p.sku, p.uploaded_at,
                   EXISTS (SELECT 1 FROM photo_reviews r WHERE r.photo_id = p.id)                  AS reviewed,
                   (SELECT COUNT(*) FROM boxes b WHERE b.photo_id = p.id AND b.status = 'accepted') AS n_accepted,
                   (SELECT COUNT(*) FROM boxes b WHERE b.photo_id = p.id AND b.status = 'proposed') AS n_proposed
            FROM photos p
            WHERE p.status = 'accepted'
            ORDER BY p.uploaded_at, p.original_name""")
        return [dict(row) for row in rows]


# ---------------------------------------------------------------- eval sets (phase 3)


def list_eval_sets() -> list[dict]:
    """All eval sets, grouped by name, newest version first, with their photo counts."""
    with closing(_connect()) as db:
        rows = db.execute("""
            SELECT e.*, (SELECT COUNT(*) FROM eval_set_photos m WHERE m.eval_set_id = e.id) AS n_photos
            FROM eval_sets e ORDER BY e.name, e.version DESC""")
        return [dict(row) for row in rows]


def get_eval_set(eval_set_id: int) -> dict | None:
    with closing(_connect()) as db:
        row = db.execute("SELECT * FROM eval_sets WHERE id = ?", (eval_set_id,)).fetchone()
    return dict(row) if row else None


def create_eval_set(name: str, version: int, category: str | None, notes: str, photo_ids: list[str]) -> int:
    """Create a draft eval set containing photo_ids. Returns its id."""
    with closing(_connect()) as db, db:
        cursor = db.execute("INSERT INTO eval_sets (name, version, category, notes, created_at) VALUES (?, ?, ?, ?, ?)",
                            (name, version, category, notes, _now()))
        eval_set_id = cursor.lastrowid
        db.executemany("INSERT INTO eval_set_photos (eval_set_id, photo_id) VALUES (?, ?)",
                       [(eval_set_id, photo_id) for photo_id in photo_ids])
    return eval_set_id


def eval_set_photo_ids(eval_set_id: int) -> list[str]:
    with closing(_connect()) as db:
        return [row[0] for row in db.execute("SELECT photo_id FROM eval_set_photos WHERE eval_set_id = ?", (eval_set_id,))]


def eval_candidates(category: str | None, not_in_set: int | None = None) -> list[str]:
    """Photos that could go into an eval set: accepted, reviewed, (if `category` is given) with at
    least one accepted box of a class in that category, and not already in eval set `not_in_set`."""
    with closing(_connect()) as db:
        rows = db.execute("""
            SELECT p.id FROM photos p
            WHERE p.status = 'accepted'
              AND EXISTS (SELECT 1 FROM photo_reviews r WHERE r.photo_id = p.id)
              AND (:category IS NULL OR EXISTS (
                    SELECT 1 FROM boxes b JOIN classes c ON c.id = b.class_id
                    WHERE b.photo_id = p.id AND b.status = 'accepted' AND c.category = :category))
              AND (:set IS NULL OR p.id NOT IN (SELECT photo_id FROM eval_set_photos WHERE eval_set_id = :set))
            ORDER BY p.uploaded_at, p.original_name""", {"category": category, "set": not_in_set})
        return [row[0] for row in rows]


def add_to_eval_set(eval_set_id: int, photo_ids: list[str]) -> None:
    with closing(_connect()) as db, db:
        db.executemany("INSERT OR IGNORE INTO eval_set_photos (eval_set_id, photo_id) VALUES (?, ?)",
                       [(eval_set_id, photo_id) for photo_id in photo_ids])


def remove_from_eval_set(eval_set_id: int, photo_id: str) -> None:
    with closing(_connect()) as db, db:
        db.execute("DELETE FROM eval_set_photos WHERE eval_set_id = ? AND photo_id = ?", (eval_set_id, photo_id))


def freeze_eval_set(eval_set_id: int) -> None:
    """Copy each photo's accepted boxes into eval_set_boxes and mark the set frozen, in one
    transaction. From then on the set's answers never change, even if a box is fixed later."""
    with closing(_connect()) as db, db:
        db.execute("""
            INSERT INTO eval_set_boxes (eval_set_id, photo_id, class_id, x1, y1, x2, y2)
            SELECT m.eval_set_id, b.photo_id, b.class_id, b.x1, b.y1, b.x2, b.y2
            FROM eval_set_photos m JOIN boxes b ON b.photo_id = m.photo_id AND b.status = 'accepted'
            WHERE m.eval_set_id = ?""", (eval_set_id,))
        db.execute("UPDATE eval_sets SET frozen_at = ? WHERE id = ?", (_now(), eval_set_id))


def delete_eval_set(eval_set_id: int) -> None:
    with closing(_connect()) as db, db:
        db.execute("DELETE FROM eval_set_photos WHERE eval_set_id = ?", (eval_set_id,))
        db.execute("DELETE FROM eval_sets WHERE id = ?", (eval_set_id,))


def eval_set_contents(eval_set_id: int, frozen: bool) -> list[dict]:
    """The set's photos, each with its boxes per class: {photo..., box_counts: {class_id: n}}.

    Frozen sets use their frozen copy of the boxes; drafts use the photos' current accepted boxes.
    """
    box_source = ("SELECT photo_id, class_id FROM eval_set_boxes WHERE eval_set_id = :set" if frozen else
                  "SELECT photo_id, class_id FROM boxes WHERE status = 'accepted' AND photo_id IN "
                  "(SELECT photo_id FROM eval_set_photos WHERE eval_set_id = :set)")
    with closing(_connect()) as db:
        photos = [dict(row) for row in db.execute("""
            SELECT p.id, p.original_name, p.sku, p.status,
                   EXISTS (SELECT 1 FROM photo_reviews r WHERE r.photo_id = p.id) AS reviewed
            FROM eval_set_photos m JOIN photos p ON p.id = m.photo_id
            WHERE m.eval_set_id = :set ORDER BY p.uploaded_at, p.original_name""", {"set": eval_set_id})]
        counts: dict[str, dict[int, int]] = {}
        for row in db.execute(box_source, {"set": eval_set_id}):
            per_class = counts.setdefault(row["photo_id"], {})
            per_class[row["class_id"]] = per_class.get(row["class_id"], 0) + 1
    for photo in photos:
        photo["box_counts"] = counts.get(photo["id"], {})
    return photos


def photos_summary(photo_ids: list[str]) -> list[dict]:
    """id, name, SKU and accepted boxes per class for some photos (for the candidates grid)."""
    if not photo_ids:
        return []
    marks = ", ".join("?" for _ in photo_ids)
    with closing(_connect()) as db:
        photos = {row["id"]: dict(row) for row in db.execute(
            f"SELECT id, original_name, sku FROM photos WHERE id IN ({marks})", photo_ids)}
        for row in db.execute(f"SELECT photo_id, class_id FROM boxes WHERE status = 'accepted' AND photo_id IN ({marks})", photo_ids):
            per_class = photos[row["photo_id"]].setdefault("box_counts", {})
            per_class[row["class_id"]] = per_class.get(row["class_id"], 0) + 1
    return [{"box_counts": {}, **photos[photo_id]} for photo_id in photo_ids if photo_id in photos]


# ---------------------------------------------------------------- dataset versions + models (phase 4)


def training_pool() -> dict:
    """Which photos a new dataset version would contain, and why others are left out.

    In: accepted + reviewed + not in any eval set. Returns {"photo_ids": [...], "in_eval_sets": n,
    "not_reviewed": n} (the two counts are accepted photos left out for that reason).
    """
    with closing(_connect()) as db:
        rows = db.execute("""
            SELECT p.id,
                   EXISTS (SELECT 1 FROM photo_reviews r WHERE r.photo_id = p.id)          AS reviewed,
                   EXISTS (SELECT 1 FROM eval_set_photos m WHERE m.photo_id = p.id)        AS in_eval
            FROM photos p WHERE p.status = 'accepted' ORDER BY p.uploaded_at, p.id""").fetchall()
    return {"photo_ids": [r["id"] for r in rows if r["reviewed"] and not r["in_eval"]],
            "in_eval_sets": sum(1 for r in rows if r["in_eval"]),
            "not_reviewed": sum(1 for r in rows if not r["reviewed"] and not r["in_eval"])}


def accepted_boxes_for(photo_ids: list[str]) -> dict[str, list[dict]]:
    """photo id -> its accepted boxes (photos without boxes map to an empty list)."""
    result = {photo_id: [] for photo_id in photo_ids}
    with closing(_connect()) as db:
        for row in db.execute("SELECT * FROM boxes WHERE status = 'accepted' ORDER BY id"):
            if row["photo_id"] in result:
                result[row["photo_id"]].append(dict(row))
    return result


def create_dataset_version(name: str, notes: str, export_path: str, class_map: dict, summary: dict,
                           splits: dict[str, str]) -> int:
    """Record a frozen dataset version and which photos went to train / val."""
    with closing(_connect()) as db, db:
        cursor = db.execute(
            "INSERT INTO dataset_versions (name, created_at, notes, export_path, class_map, summary) VALUES (?, ?, ?, ?, ?, ?)",
            (name, _now(), notes, export_path, json.dumps(class_map), json.dumps(summary)))
        db.executemany("INSERT INTO dataset_version_photos (dataset_version_id, photo_id, split) VALUES (?, ?, ?)",
                       [(cursor.lastrowid, photo_id, split) for photo_id, split in splits.items()])
        return cursor.lastrowid


def list_dataset_versions() -> list[dict]:
    """Newest first. class_map and summary are returned as dicts."""
    with closing(_connect()) as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM dataset_versions ORDER BY id DESC")]
    for row in rows:
        row["class_map"], row["summary"] = json.loads(row["class_map"]), json.loads(row["summary"])
    return rows


def add_model(fields: dict) -> None:
    with closing(_connect()) as db, db:
        db.execute("INSERT INTO models (id, path, dataset_version_id, class_names, trained_at, registered_at) "
                   "VALUES (:id, :path, :dataset_version_id, :class_names, :trained_at, :registered_at)",
                   {**fields, "registered_at": _now()})


def list_models() -> list[dict]:
    """Newest first, with the dataset version's name."""
    with closing(_connect()) as db:
        rows = [dict(row) for row in db.execute("""
            SELECT m.*, d.name AS dataset_version FROM models m
            LEFT JOIN dataset_versions d ON d.id = m.dataset_version_id ORDER BY m.trained_at DESC""")]
    for row in rows:
        row["class_names"] = json.loads(row["class_names"] or "{}")
    return rows


def set_promoted(model_id: str) -> None:
    """Make this the model the app uses (and no other)."""
    with closing(_connect()) as db, db:
        db.execute("UPDATE models SET promoted = (id = ?)", (model_id,))


# ---------------------------------------------------------------- evaluation (phase 5)


def eval_set_answers(eval_set_id: int) -> dict[str, list[dict]]:
    """photo id -> the frozen correct boxes for a frozen eval set (empty list = nothing there)."""
    with closing(_connect()) as db:
        answers = {row[0]: [] for row in db.execute("SELECT photo_id FROM eval_set_photos WHERE eval_set_id = ?", (eval_set_id,))}
        for row in db.execute("SELECT * FROM eval_set_boxes WHERE eval_set_id = ?", (eval_set_id,)):
            answers[row["photo_id"]].append(dict(row))
    return answers


def reviewed_classes(photo_ids: list[str]) -> dict[str, set[int]]:
    """photo id -> the classes it has been fully checked for (coverage)."""
    result = {photo_id: set() for photo_id in photo_ids}
    with closing(_connect()) as db:
        for row in db.execute("SELECT photo_id, class_id FROM photo_reviews"):
            if row["photo_id"] in result:
                result[row["photo_id"]].add(row["class_id"])
    return result


def save_eval_run(model_id: str, eval_set_id: int, scored_classes: list[int], metrics: dict, details: list) -> None:
    with closing(_connect()) as db, db:
        db.execute("DELETE FROM eval_runs WHERE model_id = ? AND eval_set_id = ?", (model_id, eval_set_id))
        db.execute("INSERT INTO eval_runs (model_id, eval_set_id, run_at, scored_classes, metrics, details) VALUES (?, ?, ?, ?, ?, ?)",
                   (model_id, eval_set_id, _now(), json.dumps(scored_classes), json.dumps(metrics), json.dumps(details)))


def list_eval_runs() -> list[dict]:
    """Every run's summary (without the per-photo details, which can be large)."""
    with closing(_connect()) as db:
        rows = [dict(row) for row in db.execute("SELECT id, model_id, eval_set_id, run_at, scored_classes, metrics FROM eval_runs")]
    for row in rows:
        row["scored_classes"], row["metrics"] = json.loads(row["scored_classes"]), json.loads(row["metrics"])
    return rows


def get_eval_run(run_id: int) -> dict | None:
    with closing(_connect()) as db:
        row = db.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    run = dict(row)
    for key in ("scored_classes", "metrics", "details"):
        run[key] = json.loads(run[key])
    return run
