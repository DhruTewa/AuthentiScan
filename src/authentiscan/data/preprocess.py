import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from authentiscan.schemas import ProcessedRow

RESAMPLE = {"bicubic": Image.Resampling.BICUBIC, "lanczos": Image.Resampling.LANCZOS}


def resize_and_crop(image: Image.Image, size: int, method: str) -> Image.Image:
    """Return an RGB square of side `size`: short side resized to `size`, then center-cropped."""
    image = ImageOps.exif_transpose(image)  # respect EXIF orientation
    image = image.convert("RGB")  # discard alpha channel if present
    w, h = image.size
    scale = size / min(w, h)
    new_w, new_h = max(size, round(w * scale)), max(size, round(h * scale))
    image = image.resize((new_w, new_h), RESAMPLE[method])
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return image.crop((left, top, left + size, top + size))


def process_image(args: tuple[str, str, str, str, str, int, str]) -> ProcessedRow:
    """Process one image. Never raises, failures are recorded in the row."""
    raw_root, out_root, rel, split, label, size, method = args  # strings so workers can receive them
    out_rel = f"{split}/{label}/{Path(rel).stem}.png"
    row = {
        "path": rel, "output_path": out_rel, "split": split, "label": label,
        "status": "failed", "error": "",
    }
    out = Path(out_root) / out_rel
    if out.exists():
        return ProcessedRow(**{**row, "status": "skipped"})
    try:
        with Image.open(Path(raw_root) / rel) as img:
            result = resize_and_crop(img, size, method)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")  # write then swap, so a crash never leaves a half file
        result.save(tmp, format="PNG")
        os.replace(tmp, out)
        row["status"] = "ok"
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return ProcessedRow(**row)


def preprocess_dataset(
    manifest: pd.DataFrame, raw_root: Path, out_root: Path,
    size: int, method: str, workers: int = 8,
) -> pd.DataFrame:
    """Process every manifest row in parallel. Returns one ProcessedRow per image, as a table."""
    jobs = [(str(raw_root), str(out_root), r.path, r.split, r.label, size, method)
            for r in manifest.itertuples()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(tqdm(pool.map(process_image, jobs, chunksize=64),
                         total=len(jobs), desc="Preprocessing", unit="img"))
    # explicit columns so an empty manifest still gives a table with the right header
    return pd.DataFrame([r.model_dump() for r in rows], columns=list(ProcessedRow.model_fields))
