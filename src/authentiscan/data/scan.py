import hashlib
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from authentiscan.schemas import ScanRow

IMG_EXT = {".jpg", ".png"}
SPLITS = ("train", "test")
EXIF_ORIENTATION_TAG = 274


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def scan_image(args: tuple[str, str]) -> ScanRow:
    """Scan one file. Never raises, failures are recorded in the row."""
    root, rel = args  # plain strings so they can be sent to worker processes
    path = Path(root) / rel
    split, label = Path(rel).parts[:2]
    row: dict[str, Any] = {
        "path": rel, "split": split, "label": label,
        "ext": path.suffix.lower().strip("."), "file_bytes": None,
        "format": None, "mode": None, "width": None, "height": None,
        "aspect_ratio": None, "exif_orientation": None, "jpeg_q": None,
        "decode_ok": False, "error": "", "warning": "",
        "pixel_std": None, "md5": None,
    }
    try:
        row["file_bytes"] = path.stat().st_size  # inside try so a vanished file is a failed row
        row["md5"] = file_md5(path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with Image.open(path) as img:  # header only, cheap
                row["format"] = img.format
                row["mode"] = img.mode
                row["width"], row["height"] = img.size
                row["aspect_ratio"] = max(img.size) / min(img.size)
                row["exif_orientation"] = img.getexif().get(EXIF_ORIENTATION_TAG)
                quant = getattr(img, "quantization", None)
                row["jpeg_q"] = quant[0][0] if quant and 0 in quant else None
                img.load()  # full decode, catches truncated files
                thumb = img.convert("L")
                thumb.thumbnail((64, 64))
                row["pixel_std"] = float(np.asarray(thumb).std())  # near 0 means blank
        row["decode_ok"] = True
        row["warning"] = "; ".join(str(w.message)[:100] for w in caught)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return ScanRow(**row)


def list_files(root: Path) -> list[str]:
    # only scan the split folders, so data/processed and data/interim are ignored
    return sorted(
        str(p.relative_to(root))
        for split in SPLITS
        for p in (root / split).rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXT
    )


def scan_dataset(root, out_csv, workers: int = 8, limit: int | None = None,
                 seed: int = 42, force: bool = False) -> pd.DataFrame:
    """Scan every image into one table. With limit, scans a random sample and writes nothing."""
    root, out_csv = Path(root), Path(out_csv)
    if limit is None and out_csv.exists() and not force:
        return pd.read_csv(out_csv)

    files = list_files(root)
    if limit is not None:
        files = sorted(random.Random(seed).sample(files, k=min(limit, len(files))))

    jobs = [(str(root), rel) for rel in files]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(scan_image, jobs, chunksize=64), 1):
            rows.append(row)
            if i % 5000 == 0:
                print(f"{i}/{len(jobs)} scanned", flush=True)

    df = pd.DataFrame([r.model_dump() for r in rows]).sort_values("path").reset_index(drop=True)
    if limit is None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df
