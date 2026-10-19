from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from authentiscan.data.dataset import AuthentiScanDataset
from authentiscan.data.stats import NormStats
from authentiscan.data.transforms import build_transform

CLASSES = {"fake": 1, "real": 0}
STATS = NormStats(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25], n_images=1)
SPLITS = {"train": 4, "val": 2, "test": 2}  # images per label


@pytest.fixture
def processed(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    rng = np.random.default_rng(0)
    rows = []
    for split, n in SPLITS.items():
        for label in CLASSES:
            for i in range(n):
                rel = f"{split}/{label}/{i:03d}.png"
                (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
                arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
                Image.fromarray(arr).save(tmp_path / rel)
                rows.append({"output_path": rel, "split": split, "label": label})
    return pd.DataFrame(rows), tmp_path


def make(processed: tuple[pd.DataFrame, Path], split: str, train: bool = False) -> AuthentiScanDataset:
    manifest, root = processed
    return AuthentiScanDataset(manifest, split, root, CLASSES, build_transform(train, 224, STATS))


@pytest.mark.parametrize("split", SPLITS)
def test_length_matches_the_split(processed: tuple[pd.DataFrame, Path], split: str) -> None:
    assert len(make(processed, split)) == SPLITS[split] * 2


def test_item_is_a_normalized_tensor_and_an_int_label(processed: tuple[pd.DataFrame, Path]) -> None:
    tensor, label = make(processed, "train")[0]
    assert tensor.shape == (3, 224, 224) and tensor.dtype == torch.float32
    assert label in (0, 1)


def test_labels_follow_the_class_mapping(processed: tuple[pd.DataFrame, Path]) -> None:
    ds = make(processed, "train")
    manifest, _ = processed
    expected = [CLASSES[c] for c in manifest[manifest["split"] == "train"]["label"]]
    assert ds.labels == expected
    assert sorted(set(ds.labels)) == [0, 1]


def test_split_only_contains_its_own_images(processed: tuple[pd.DataFrame, Path]) -> None:
    ds = make(processed, "val")
    assert all(p.parts[-3] == "val" for p in ds.paths)


def test_unknown_split_raises(processed: tuple[pd.DataFrame, Path]) -> None:
    with pytest.raises(ValueError, match="training"):
        make(processed, "training")


def test_unknown_label_raises_at_construction(processed: tuple[pd.DataFrame, Path]) -> None:
    manifest, root = processed
    bad = manifest.copy()
    bad.loc[0, "label"] = "cat"
    with pytest.raises(KeyError):
        AuthentiScanDataset(bad, "train", root, CLASSES, build_transform(False, 224, STATS))


def test_missing_file_raises_when_fetched(processed: tuple[pd.DataFrame, Path]) -> None:
    # no silent skipping, a missing file means something is really wrong
    ds = make(processed, "train")
    ds.paths[0].unlink()
    with pytest.raises(FileNotFoundError):
        ds[0]


def test_eval_items_are_repeatable_and_train_items_vary(processed: tuple[pd.DataFrame, Path]) -> None:
    ev = make(processed, "train", train=False)
    assert torch.equal(ev[0][0], ev[0][0])
    tr = make(processed, "train", train=True)
    assert any(not torch.equal(tr[0][0], tr[0][0]) for _ in range(5))


@pytest.mark.parametrize("workers", [0, 2])
def test_dataloader_batches(processed: tuple[pd.DataFrame, Path], workers: int) -> None:
    loader = DataLoader(make(processed, "train", train=True), batch_size=4, shuffle=False,
                        num_workers=workers)
    images, labels = next(iter(loader))
    assert images.shape == (4, 3, 224, 224)
    assert labels.dtype == torch.int64 and labels.shape == (4,)
    assert sum(len(y) for _, y in loader) == 8
