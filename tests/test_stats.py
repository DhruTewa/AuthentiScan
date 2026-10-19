from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from authentiscan.data.stats import compute_norm_stats


def save(root: Path, rel: str, colour: tuple[int, int, int], size: tuple[int, int] = (4, 4)) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)


def manifest(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"output_path": p, "split": s} for p, s in rows])


def test_single_flat_image_has_its_value_as_mean_and_zero_std(tmp_path: Path) -> None:
    save(tmp_path, "train/real/a.png", (255, 0, 51))
    stats = compute_norm_stats(manifest([("train/real/a.png", "train")]), tmp_path)
    assert stats.mean == pytest.approx([1.0, 0.0, 0.2])
    assert stats.std == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert stats.n_images == 1


def test_black_and_white_pair_matches_hand_calculation(tmp_path: Path) -> None:
    # half the pixels 0 and half 1: mean 0.5, std 0.5 in every channel
    save(tmp_path, "train/real/black.png", (0, 0, 0))
    save(tmp_path, "train/fake/white.png", (255, 255, 255))
    m = manifest([("train/real/black.png", "train"), ("train/fake/white.png", "train")])
    stats = compute_norm_stats(m, tmp_path)
    assert stats.mean == pytest.approx([0.5] * 3)
    assert stats.std == pytest.approx([0.5] * 3)
    assert stats.n_images == 2


def test_only_train_rows_are_used(tmp_path: Path) -> None:
    save(tmp_path, "train/real/a.png", (0, 0, 0))
    save(tmp_path, "val/real/b.png", (255, 255, 255))
    save(tmp_path, "test/real/c.png", (255, 255, 255))
    m = manifest([
        ("train/real/a.png", "train"), ("val/real/b.png", "val"), ("test/real/c.png", "test"),
    ])
    stats = compute_norm_stats(m, tmp_path)
    assert stats.mean == pytest.approx([0.0] * 3)
    assert stats.n_images == 1


def test_matches_numpy_on_random_images(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    arrays = [rng.integers(0, 256, (8, 8, 3), dtype=np.uint8) for _ in range(5)]
    rows = []
    for i, arr in enumerate(arrays):
        rel = f"train/real/{i}.png"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(tmp_path / rel)
        rows.append((rel, "train"))
    stats = compute_norm_stats(manifest(rows), tmp_path)
    stacked = np.stack(arrays).astype(np.float64) / 255.0
    assert stats.mean == pytest.approx(stacked.mean(axis=(0, 1, 2)))
    assert stats.std == pytest.approx(stacked.std(axis=(0, 1, 2)))


def test_missing_image_raises(tmp_path: Path) -> None:
    # unlike preprocessing, stats must not quietly skip a file, or the numbers would be wrong
    with pytest.raises(FileNotFoundError):
        compute_norm_stats(manifest([("train/real/ghost.png", "train")]), tmp_path)
