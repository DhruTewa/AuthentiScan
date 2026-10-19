from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm


class NormStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mean: list[float]
    std: list[float]
    n_images: int


def compute_norm_stats(manifest: pd.DataFrame, processed_dir: Path) -> NormStats:
    """Per-channel mean and std over the train split only, pixel values scaled to 0..1."""
    train = manifest[manifest["split"] == "train"]
    total = np.zeros(3)  # float64 running sums, so precision holds over billions of pixels
    total_sq = np.zeros(3)
    n_pixels = 0

    for rel in tqdm(train["output_path"], desc="Computing stats", unit="img"):
        with Image.open(processed_dir / rel) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        total += arr.sum(axis=(0, 1))
        total_sq += (arr**2).sum(axis=(0, 1))
        n_pixels += arr.shape[0] * arr.shape[1]

    mean = total / n_pixels
    std = np.sqrt(total_sq / n_pixels - mean**2)  # variance = E[x^2] - mean^2
    return NormStats(mean=mean.tolist(), std=std.tolist(), n_images=len(train))