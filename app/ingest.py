"""The upload pipeline: ingest → normalize → check (design.md §6).

ingest() runs every step for one uploaded file. Step numbers match the table in design.md §6.0,
so the two can be read side by side.

Result for each upload, stored as `status` in the catalog:
    accepted   passed every check; will be used for labeling and training
    rejected   can't be decoded, too small, or a near-duplicate; `check_notes` says why
Rejected photos keep their original file (D3), so nothing is ever lost.
"""

import hashlib
import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps
from pillow_heif import register_heif_opener

from app import catalog, storage
from app.settings import JPEG_QUALITY, MIN_SHORT_SIDE, NEAR_DUPLICATE_BITS, THUMB_SIZE

register_heif_opener()  # teaches Pillow to open iPhone .heic/.heif files

SRGB = ImageCms.createProfile("sRGB")  # the standard color space of the web and of most cameras

# EXIF tag numbers (the EXIF standard identifies fields by number, not name)
EXIF_MAKE, EXIF_MODEL, EXIF_DATETIME = 0x010F, 0x0110, 0x0132
EXIF_SUB_IFD, EXIF_DATETIME_ORIGINAL = 0x8769, 0x9003


def ingest(data: bytes, original_name: str, sku: str, uploaded_by: str, upload_batch: str) -> tuple[dict, bool]:
    """Run the pipeline on one uploaded file.

    Returns (the photo's catalog row, already_uploaded). already_uploaded is True when this
    exact file was uploaded before; then nothing new is stored.
    """
    # ---- Ingest (§6.2): on the original bytes, before anything is changed ----

    # 1. SHA-256 fingerprint of the exact bytes. It becomes the photo's id.
    photo_id = hashlib.sha256(data).hexdigest()

    # 2. Exact-duplicate check: the same file was uploaded before -> keep the existing photo.
    existing = catalog.get_photo(photo_id)
    if existing:
        return existing, True

    # 3. Store the original exactly as received (D3).
    original_key = f"originals/{photo_id}{Path(original_name).suffix.lower()}"
    storage.save(original_key, data)

    # 4. Catalog row. Status stays "processing" until the checks below have run.
    catalog.add_photo({
        "id": photo_id, "original_name": original_name, "original_key": original_key,
        "uploaded_by": uploaded_by, "upload_batch": upload_batch,
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
        "sku": sku or None, "status": "processing",
    })

    # ---- Normalize (§6.3): produce processed/<id>.jpg ----

    # 5. Decode the file into pixels (JPEG, PNG, WebP, HEIC...).
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:  # not an image, an unsupported format, or a damaged file
        return _finish(photo_id, {"status": "rejected", "check_notes": "can't be read as an image (unknown or damaged file)"})

    # 6. Copy useful metadata into the catalog. The color profile is kept for step 8.
    device, taken_at = _camera_info(image)
    color_profile = image.info.get("icc_profile")

    # 7. Rotate: phones often store pixels sideways plus an EXIF tag saying how to turn them.
    #    This applies the turn to the pixels. (HEIC files are already turned by the decoder.)
    image = ImageOps.exif_transpose(image)

    # 8. Convert colors to sRGB. iPhones shoot in "Display P3", a wider color space; without
    #    this, the same red would be stored as different numbers in different photos.
    image = _to_srgb(image, color_profile)

    # 9. Plain 8-bit RGB: drops transparency, turns grayscale/CMYK/16-bit images into RGB.
    image = image.convert("RGB")

    # 10. (No resize, D10: the processed photo keeps full resolution.)

    # 11 + 12. Encode as JPEG. Pillow writes no EXIF, GPS or color-profile data unless told
    #          to, so this also strips all metadata from the processed copy (privacy).
    processed_key = f"processed/{photo_id}.jpg"
    storage.save(processed_key, _jpeg_bytes(image, JPEG_QUALITY))

    # 12a. Thumbnail, used only to show the photo on web pages (never for labeling/training).
    thumbnail = image.copy()
    thumbnail.thumbnail((THUMB_SIZE, THUMB_SIZE))  # shrinks to fit, keeping the shape
    thumb_key = f"thumbs/{photo_id}.jpg"
    storage.save(thumb_key, _jpeg_bytes(thumbnail, 85))

    facts = {"processed_key": processed_key, "thumb_key": thumb_key, "device": device, "taken_at": taken_at,
             "width": image.width, "height": image.height}

    # ---- Check (§6.4): on the processed image ----

    # 13. Too small to be useful?
    if min(image.size) < MIN_SHORT_SIDE:
        return _finish(photo_id, {**facts, "status": "rejected",
                                  "check_notes": f"too small: {image.width}x{image.height}px (short side must be ≥ {MIN_SHORT_SIDE}px)"})

    # 14. Perceptual hash: a fingerprint of what the photo looks like.
    phash = perceptual_hash(image)
    facts["phash"] = phash

    # 15. Near-duplicate check (D9): does it look almost the same as an accepted photo?
    for other_id, other_hash in catalog.accepted_phashes():
        differing_bits = (int(phash, 16) ^ int(other_hash, 16)).bit_count()
        if differing_bits <= NEAR_DUPLICATE_BITS:
            return _finish(photo_id, {**facts, "status": "rejected", "duplicate_of": other_id,
                                      "check_notes": f"near-duplicate of an earlier photo ({differing_bits} of 64 bits differ)"})

    return _finish(photo_id, {**facts, "status": "accepted"})


def perceptual_hash(image: Image.Image) -> str:
    """64-bit perceptual hash ("pHash") as 16 hex digits.

    Shrink to a 32x32 grayscale image, take its 2D frequency spectrum (DCT) and keep the 8x8
    lowest frequencies, which describe the image's broad light/dark layout. Each bit = is that
    frequency above the median. Re-saving, recompressing or slightly resizing a photo barely
    changes its layout, so the hash stays (nearly) the same; a different scene changes it a lot.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    low_frequencies = cv2.dct(small)[:8, :8].flatten()
    bits = low_frequencies > np.median(low_frequencies)
    return f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"


def _camera_info(image: Image.Image) -> tuple[str | None, str | None]:
    """(device, taken_at) from EXIF, e.g. ("Apple iPhone 15", "2026:10:14 17:30:05")."""
    exif = image.getexif()
    device = " ".join(str(exif[tag]).strip() for tag in (EXIF_MAKE, EXIF_MODEL) if exif.get(tag)) or None
    taken_at = exif.get_ifd(EXIF_SUB_IFD).get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
    return device, str(taken_at) if taken_at else None


def _to_srgb(image: Image.Image, color_profile: bytes | None) -> Image.Image:
    """Convert to sRGB using the photo's embedded color profile.

    No profile = the photo is assumed to be sRGB already (the convention for untagged images).
    A profile that can't be applied (rare formats) leaves the colors as they are.
    """
    if not color_profile:
        return image
    try:
        source = ImageCms.ImageCmsProfile(io.BytesIO(color_profile))
        return ImageCms.profileToProfile(image, source, SRGB, outputMode="RGB")
    except (ImageCms.PyCMSError, OSError, ValueError):
        return image


def _jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def _finish(photo_id: str, fields: dict) -> tuple[dict, bool]:
    """Write the outcome to the catalog and return the final row."""
    catalog.update_photo(photo_id, fields)
    return catalog.get_photo(photo_id), False
