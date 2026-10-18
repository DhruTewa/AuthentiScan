import pandas as pd
import pytest

from authentiscan.data.split import assign_val_split


@pytest.fixture
def manifest() -> pd.DataFrame:
    rows = [
        {"path": f"{split}/{label}/{i:03d}.png", "split": split, "label": label}
        for split, n in (("train", 100), ("test", 20))
        for label in ("fake", "real")
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def val_paths(df: pd.DataFrame) -> set[str]:
    return set(df.loc[df["split"] == "val", "path"])


def test_val_is_a_tenth_of_each_label(manifest: pd.DataFrame) -> None:
    out = assign_val_split(manifest, val_fraction=0.1, seed=42)
    val = out[out["split"] == "val"]
    assert val["label"].value_counts().to_dict() == {"fake": 10, "real": 10}


def test_only_train_rows_become_val(manifest: pd.DataFrame) -> None:
    out = assign_val_split(manifest, val_fraction=0.1, seed=42)
    assert all(p.startswith("train/") for p in val_paths(out))
    assert (out[out["split"] == "test"]["path"].str.startswith("test/")).all()
    assert (out["split"] == "test").sum() == 40


def test_row_count_and_other_columns_unchanged(manifest: pd.DataFrame) -> None:
    out = assign_val_split(manifest, val_fraction=0.1, seed=42)
    assert len(out) == len(manifest)
    assert set(out["path"]) == set(manifest["path"])
    assert out.set_index("path")["label"].equals(manifest.set_index("path")["label"].reindex(out["path"]))


def test_same_seed_gives_same_val(manifest: pd.DataFrame) -> None:
    a = assign_val_split(manifest, val_fraction=0.1, seed=42)
    b = assign_val_split(manifest, val_fraction=0.1, seed=42)
    assert val_paths(a) == val_paths(b)


def test_row_order_does_not_change_the_split(manifest: pd.DataFrame) -> None:
    shuffled = manifest.sample(frac=1.0, random_state=1)
    a = assign_val_split(manifest, val_fraction=0.1, seed=42)
    b = assign_val_split(shuffled, val_fraction=0.1, seed=42)
    assert val_paths(a) == val_paths(b)


def test_different_seed_gives_different_val(manifest: pd.DataFrame) -> None:
    a = assign_val_split(manifest, val_fraction=0.1, seed=42)
    b = assign_val_split(manifest, val_fraction=0.1, seed=7)
    assert val_paths(a) != val_paths(b)


def test_input_is_not_modified(manifest: pd.DataFrame) -> None:
    before = manifest.copy()
    assign_val_split(manifest, val_fraction=0.1, seed=42)
    pd.testing.assert_frame_equal(manifest, before)


def test_not_safe_to_apply_to_its_own_output(manifest: pd.DataFrame) -> None:
    # documents a known limit: a second pass takes another 10% of the remaining train rows,
    # so always feed it the filtered manifest, never a manifest that already has val
    once = assign_val_split(manifest, val_fraction=0.1, seed=42)
    twice = assign_val_split(once, val_fraction=0.1, seed=42)
    assert (twice["split"] == "val").sum() > (once["split"] == "val").sum()
