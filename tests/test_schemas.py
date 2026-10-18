from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from authentiscan.schemas import DataConfig, ScanRow, load_data_config

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "data.yaml"


@pytest.fixture
def raw_config() -> dict[str, Any]:
    # fresh dict per test so one test's edits never leak into another
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_real_config_loads() -> None:
    config = load_data_config(CONFIG_PATH)
    assert config.preprocess.output_format == "PNG"


@pytest.mark.parametrize(
    "section, key, value",
    [
        ("preprocess", "output_format", "JPEG"),
        ("preprocess", "resize_method", "nearest"),
        ("preprocess", "image_size", 0),
        ("split", "val_fraction", 1.5),
        ("split", "val_fraction", 0),
        ("filters", "min_short_side", 0),
        ("filters", "max_aspect_ratio", 1),
        ("filters", "min_pixel_std", -1),
        ("filters", "dedupe_exact", "maybe"),
    ],
)
def test_rejects_bad_values(
    raw_config: dict[str, Any], section: str, key: str, value: Any
) -> None:
    raw_config[section][key] = value
    with pytest.raises(ValidationError):
        DataConfig.model_validate(raw_config)


def test_rejects_unknown_key(raw_config: dict[str, Any]) -> None:
    raw_config["preprocess"]["image_sise"] = 256
    with pytest.raises(ValidationError):
        DataConfig.model_validate(raw_config)


def test_rejects_missing_section(raw_config: dict[str, Any]) -> None:
    raw_config.pop("split")
    with pytest.raises(ValidationError):
        DataConfig.model_validate(raw_config)


def test_rejects_upscaling(raw_config: dict[str, Any]) -> None:
    raw_config["filters"]["min_short_side"] = 200
    with pytest.raises(ValidationError):
        DataConfig.model_validate(raw_config)


def test_rejects_identical_class_labels(raw_config: dict[str, Any]) -> None:
    raw_config["classes"] = {"fake": 0, "real": 0}
    with pytest.raises(ValidationError):
        DataConfig.model_validate(raw_config)


def test_real_config_has_scan_and_rejected_paths() -> None:
    paths = load_data_config(CONFIG_PATH).paths
    assert paths.scan_path == Path("data/interim/scan.csv")
    assert paths.manifest_path == Path("data/interim/dataset_manifest.csv")
    assert paths.rejected_path == Path("data/interim/excluded_images.csv")
    assert paths.split_manifest_path == Path("data/interim/split_manifest.csv")


@pytest.mark.parametrize("key", ["scan_path", "rejected_path", "split_manifest_path"])
def test_rejects_missing_path_key(raw_config: dict[str, Any], key: str) -> None:
    raw_config["paths"].pop(key)
    with pytest.raises(ValidationError):
        DataConfig.model_validate(raw_config)


def test_rejects_unknown_path_key(raw_config: dict[str, Any]) -> None:
    # catches the rejects_path vs rejected_path kind of typo
    raw_config["paths"]["rejects_path"] = "data/interim/rejects.csv"
    with pytest.raises(ValidationError):
        DataConfig.model_validate(raw_config)


def test_scan_row_defaults_to_failed() -> None:
    row = ScanRow(path="train/real/a.png", split="train", label="real", ext="png")
    assert not row.decode_ok
    assert row.width is None and row.md5 is None
    assert row.error == "" and row.warning == ""


def test_scan_row_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ScanRow(path="a", split="train", label="real", ext="png", colour="red")  # type: ignore[call-arg]


def test_scan_row_needs_identity_fields() -> None:
    with pytest.raises(ValidationError):
        ScanRow(path="a", split="train", label="real")  # type: ignore[call-arg]
