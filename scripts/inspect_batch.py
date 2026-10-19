import logging
from pathlib import Path

import matplotlib
import pandas as pd
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")  # save to a file, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from authentiscan.data.dataset import AuthentiScanDataset  # noqa: E402
from authentiscan.data.stats import NormStats  # noqa: E402
from authentiscan.data.transforms import build_transform, denormalize  # noqa: E402
from authentiscan.schemas import load_data_config  # noqa: E402

CONFIG_PATH = Path("configs/data.yaml")
OUT_PATH = Path("data/interim/batch_check.png")  # data/ is gitignored


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_data_config(CONFIG_PATH)
    manifest = pd.read_csv(cfg.paths.processed_manifest_path)
    stats = NormStats.model_validate_json(cfg.paths.norm_stats_path.read_text())
    class_to_idx = cfg.classes.model_dump()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    dataset = AuthentiScanDataset(
        manifest, "train", cfg.paths.processed_dir, class_to_idx,
        build_transform(True, cfg.preprocess.crop_size, stats),
    )
    loader = DataLoader(
        dataset, batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(cfg.split.seed),  # same batch every run
    )
    images, labels = next(iter(loader))
    logging.info("shape %s dtype %s min %.2f max %.2f",
                 tuple(images.shape), images.dtype, images.min(), images.max())
    logging.info("fake %d, real %d", (labels == 1).sum(), (labels == 0).sum())

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for ax, img, y in zip(axes.flat, denormalize(images, stats), labels):
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
        ax.set_title(idx_to_class[int(y)])
        ax.axis("off")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=100)
    logging.info("saved %s", OUT_PATH)


if __name__ == "__main__":
    main()
