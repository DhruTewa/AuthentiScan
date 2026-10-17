from typing import Any

import pandas as pd
import pytest

from authentiscan.data.filters import apply_filters
from authentiscan.schemas import FiltersConfig

NAN = float("nan")


@pytest.fixture
def cfg() -> FiltersConfig:
    # explicit values so tests do not change when data.yaml is tuned
    return FiltersConfig(
        min_short_side=256, max_aspect_ratio=3.0, min_pixel_std=2.0, dedupe_exact=True
    )


def make_row(path: str, **overrides: Any) -> dict[str, Any]:
    # a healthy image by default, each test breaks one thing
    row: dict[str, Any] = {
        "path": path,
        "split": path.split("/")[0],
        "label": "real",
        "width": 512,
        "height": 512,
        "aspect_ratio": 1.0,
        "pixel_std": 40.0,
        "decode_ok": True,
        "md5": f"md5-{path}",
    }
    row.update(overrides)
    return row


def run(
    rows: list[dict[str, Any]], cfg: FiltersConfig
) -> tuple[set[str], dict[str, str]]:
    """Return (kept paths, {rejected path: reason})."""
    kept, rejects = apply_filters(pd.DataFrame(rows), cfg)
    return set(kept["path"]), dict(zip(rejects["path"], rejects["reject_reason"]))


def test_healthy_row_is_kept(cfg: FiltersConfig) -> None:
    kept, rejected = run([make_row("train/real/a.jpg")], cfg)
    assert kept == {"train/real/a.jpg"}
    assert rejected == {}


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (
            {"decode_ok": False, "width": NAN, "height": NAN,
             "aspect_ratio": NAN, "pixel_std": NAN},
            "decode_failed",
        ),
        ({"width": 200}, "too_small"),
        ({"height": 255}, "too_small"),
        ({"aspect_ratio": 3.5}, "extreme_aspect"),
        ({"pixel_std": 1.0}, "low_contrast"),
    ],
)
def test_each_rule_rejects_with_its_reason(
    cfg: FiltersConfig, overrides: dict[str, Any], reason: str
) -> None:
    kept, rejected = run([make_row("train/real/a.jpg", **overrides)], cfg)
    assert kept == set()
    assert rejected == {"train/real/a.jpg": reason}


def test_limits_are_inclusive_at_the_boundary(cfg: FiltersConfig) -> None:
    # exactly on the limit stays: min size 256, aspect 3.0, std 2.0
    row = make_row("train/real/a.jpg", width=256, height=256, aspect_ratio=3.0, pixel_std=2.0)
    kept, rejected = run([row], cfg)
    assert kept == {"train/real/a.jpg"}
    assert rejected == {}


def test_row_breaking_two_rules_lists_both_reasons(cfg: FiltersConfig) -> None:
    row = make_row("train/real/a.jpg", width=200, pixel_std=1.0)
    _, rejected = run([row], cfg)
    assert set(rejected["train/real/a.jpg"].split(";")) == {"too_small", "low_contrast"}


def test_test_copy_of_a_train_file_is_dropped(cfg: FiltersConfig) -> None:
    rows = [
        make_row("train/real/a.jpg", md5="same"),
        make_row("test/real/a.jpg", md5="same"),
    ]
    kept, rejected = run(rows, cfg)
    assert kept == {"train/real/a.jpg"}
    assert rejected == {"test/real/a.jpg": "duplicate_exact"}


def test_duplicates_inside_one_split_keep_the_first_path(cfg: FiltersConfig) -> None:
    rows = [
        make_row("train/real/b.jpg", md5="same"),
        make_row("train/real/a.jpg", md5="same"),
    ]
    kept, rejected = run(rows, cfg)
    assert kept == {"train/real/a.jpg"}
    assert rejected == {"train/real/b.jpg": "duplicate_exact"}


def test_rejected_copy_does_not_count_as_the_surviving_duplicate(cfg: FiltersConfig) -> None:
    # the train copy is too small, so the test copy is the only usable one
    rows = [
        make_row("train/real/a.jpg", md5="same", width=200),
        make_row("test/real/a.jpg", md5="same"),
    ]
    kept, rejected = run(rows, cfg)
    assert kept == {"test/real/a.jpg"}
    assert rejected == {"train/real/a.jpg": "too_small"}


def test_dedupe_can_be_switched_off(cfg: FiltersConfig) -> None:
    rows = [
        make_row("train/real/a.jpg", md5="same"),
        make_row("test/real/a.jpg", md5="same"),
    ]
    kept, rejected = run(rows, cfg.model_copy(update={"dedupe_exact": False}))
    assert kept == {"train/real/a.jpg", "test/real/a.jpg"}
    assert rejected == {}


def test_every_row_is_either_kept_or_rejected_once(cfg: FiltersConfig) -> None:
    rows = [
        make_row("train/real/a.jpg"),
        make_row("train/real/b.jpg", width=100),
        make_row("train/fake/c.jpg", md5="same"),
        make_row("test/fake/c.jpg", md5="same"),
    ]
    kept, rejected = run(rows, cfg)
    assert kept | set(rejected) == {r["path"] for r in rows}
    assert kept & set(rejected) == set()


def test_input_order_does_not_change_the_result(cfg: FiltersConfig) -> None:
    rows = [
        make_row("train/real/a.jpg", md5="same"),
        make_row("train/real/b.jpg", md5="same"),
        make_row("train/fake/c.jpg", pixel_std=0.5),
        make_row("test/real/d.jpg"),
    ]
    assert run(rows, cfg) == run(list(reversed(rows)), cfg)


def test_input_table_is_not_modified(cfg: FiltersConfig) -> None:
    df = pd.DataFrame([make_row("train/real/a.jpg"), make_row("train/real/b.jpg", width=100)])
    before = df.copy()
    apply_filters(df, cfg)
    pd.testing.assert_frame_equal(df, before)


def test_rerunning_on_the_kept_rows_rejects_nothing(cfg: FiltersConfig) -> None:
    # idempotent: a clean table passes through unchanged
    rows = [
        make_row("train/real/a.jpg", md5="same"),
        make_row("test/real/a.jpg", md5="same"),
        make_row("train/fake/c.jpg", width=100),
        make_row("train/fake/d.jpg"),
    ]
    kept, _ = apply_filters(pd.DataFrame(rows), cfg)
    kept_again, rejects_again = apply_filters(kept, cfg)
    assert len(rejects_again) == 0
    assert set(kept_again["path"]) == set(kept["path"])
