from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class AuthentiScanDataset(Dataset):
    """Loads processed images for one split and returns (normalized tensor, integer label)."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        processed_dir: Path,
        class_to_idx: dict[str, int],
        transform: v2.Compose,
    ) -> None:
        rows = manifest[manifest["split"] == split]
        if rows.empty:
            raise ValueError(f"no rows for split {split!r}")  # catches a typo like "training"
        # plain lists are faster per item than pandas and copy cleanly to worker processes
        self.paths = [processed_dir / p for p in rows["output_path"]]
        self.labels = [class_to_idx[c] for c in rows["label"]]  # unknown label fails here, early
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[idx]) as img:
            tensor = self.transform(img.convert("RGB"))  # transform runs while the file is open
        return tensor, self.labels[idx]
