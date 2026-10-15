"""Eval sets: named, versioned groups of photos held back to grade models (design.md §7.1, D5).

Life of an eval set:
    1. create        a draft "shoes-eval" v1 = a random X% of the eligible photos
    2. edit (draft)  remove photos, add specific ones, or add N more at random
    3. freeze        the photos AND their current boxes are copied and locked forever
    4. new version   later, copy it into a draft v2, edit, freeze

Eligible photos: accepted, reviewed, and (if the set has a category, e.g. footwear) with at
least one accepted box of a class in that category. Only reviewed photos count, because an
eval set's boxes are the correct answers.

Photos in any eval set (draft or frozen) are kept out of training (phase 4).
"""

import random

from app import catalog


class EvalSetError(ValueError):
    """A request that breaks the rules above; the message explains why."""


def create(name: str, category: str | None, percent: float, notes: str) -> int:
    """New eval set name -> draft v1 with a random `percent`% of the eligible photos."""
    if any(e["name"] == name for e in catalog.list_eval_sets()):
        raise EvalSetError(f"'{name}' already exists. Make a new version of it instead.")
    pool = catalog.eval_candidates(category)
    count = max(1, round(len(pool) * percent / 100)) if pool else 0
    return catalog.create_eval_set(name, 1, category, notes, random.sample(pool, count))


def new_version(eval_set_id: int) -> int:
    """Copy an eval set into a new draft version (latest version + 1) with the same photos."""
    source = _get(eval_set_id)
    latest = max(e["version"] for e in catalog.list_eval_sets() if e["name"] == source["name"])
    return catalog.create_eval_set(source["name"], latest + 1, source["category"], source["notes"] or "",
                                   catalog.eval_set_photo_ids(eval_set_id))


def add_random(eval_set_id: int, count: int) -> int:
    """Add up to `count` random eligible photos to a draft. Returns how many were added."""
    eval_set = _draft(eval_set_id)
    pool = catalog.eval_candidates(eval_set["category"], not_in_set=eval_set_id)
    chosen = random.sample(pool, min(count, len(pool)))
    catalog.add_to_eval_set(eval_set_id, chosen)
    return len(chosen)


def add_photos(eval_set_id: int, photo_ids: list[str]) -> None:
    """Add specific photos (picked from the candidates) to a draft."""
    eval_set = _draft(eval_set_id)
    eligible = set(catalog.eval_candidates(eval_set["category"], not_in_set=eval_set_id))
    if not set(photo_ids) <= eligible:
        raise EvalSetError("Only accepted, reviewed photos from the candidates list can be added.")
    catalog.add_to_eval_set(eval_set_id, photo_ids)


def remove_photo(eval_set_id: int, photo_id: str) -> None:
    _draft(eval_set_id)
    catalog.remove_from_eval_set(eval_set_id, photo_id)


def freeze(eval_set_id: int) -> None:
    """Lock a draft. Refused if it's empty or holds photos that are no longer usable."""
    _draft(eval_set_id)
    photos = catalog.eval_set_contents(eval_set_id, frozen=False)
    if not photos:
        raise EvalSetError("An empty eval set can't be frozen.")
    unusable = [p["original_name"] for p in photos if p["status"] != "accepted" or not p["reviewed"]]
    if unusable:
        raise EvalSetError(f"Remove or review these first (rejected or not reviewed): {', '.join(unusable[:5])}"
                           + (" …" if len(unusable) > 5 else ""))
    catalog.freeze_eval_set(eval_set_id)


def delete(eval_set_id: int) -> None:
    """Delete a draft. Frozen sets are permanent (models get scored against them)."""
    _draft(eval_set_id)
    catalog.delete_eval_set(eval_set_id)


def details(eval_set_id: int) -> dict:
    """Everything the Eval sets page shows for one set."""
    eval_set = _get(eval_set_id)
    frozen = bool(eval_set["frozen_at"])
    candidates = [] if frozen else catalog.photos_summary(catalog.eval_candidates(eval_set["category"], not_in_set=eval_set_id))
    return {"eval_set": eval_set, "photos": catalog.eval_set_contents(eval_set_id, frozen), "candidates": candidates}


def _get(eval_set_id: int) -> dict:
    eval_set = catalog.get_eval_set(eval_set_id)
    if eval_set is None:
        raise EvalSetError("No such eval set.")
    return eval_set


def _draft(eval_set_id: int) -> dict:
    eval_set = _get(eval_set_id)
    if eval_set["frozen_at"]:
        raise EvalSetError("This version is frozen and can't change. Make a new version instead.")
    return eval_set
