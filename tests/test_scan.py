import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from authentiscan.data.scan import list_files, scan_dataset, scan_image
from authentiscan.schemas import ScanRow


def noise_image(size: tuple[int, int] = (64, 64), mode: str = "RGB") -> Image.Image:
    # seeded noise so every run produces the same bytes
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr).convert(mode)


def save(root: Path, rel: str, img: Image.Image, **kwargs: object) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, **kwargs)
    return path


def scan(root: Path, rel: str) -> ScanRow:
    return scan_image((str(root), rel))


def test_healthy_png(tmp_path: Path) -> None:
    path = save(tmp_path, "train/real/a.png", noise_image((80, 40)))
    row = scan(tmp_path, "train/real/a.png")
    assert row.decode_ok and row.error == ""
    assert (row.split, row.label, row.ext) == ("train", "real", "png")
    assert (row.format, row.mode) == ("PNG", "RGB")
    assert (row.width, row.height, row.aspect_ratio) == (80, 40, 2.0)
    assert row.file_bytes == path.stat().st_size
    assert row.md5 == hashlib.md5(path.read_bytes()).hexdigest()
    assert row.pixel_std is not None and row.pixel_std > 10
    assert row.jpeg_q is None


def test_jpeg_has_quant_table(tmp_path: Path) -> None:
    save(tmp_path, "train/real/a.jpg", noise_image(), quality=90)
    row = scan(tmp_path, "train/real/a.jpg")
    assert row.format == "JPEG"
    assert row.jpeg_q is not None


def test_format_comes_from_content_not_extension(tmp_path: Path) -> None:
    save(tmp_path, "train/fake/a.jpg", noise_image(), format="PNG")
    row = scan(tmp_path, "train/fake/a.jpg")
    assert row.ext == "jpg"
    assert row.format == "PNG"


def test_palette_mode(tmp_path: Path) -> None:
    save(tmp_path, "train/fake/a.png", noise_image().convert("P"))
    assert scan(tmp_path, "train/fake/a.png").mode == "P"


def test_portrait_aspect_ratio_is_at_least_one(tmp_path: Path) -> None:
    save(tmp_path, "test/real/a.png", noise_image((50, 150)))
    assert scan(tmp_path, "test/real/a.png").aspect_ratio == 3.0


def test_blank_image_has_zero_std(tmp_path: Path) -> None:
    save(tmp_path, "train/fake/a.png", Image.new("RGB", (64, 64), (128, 128, 128)))
    row = scan(tmp_path, "train/fake/a.png")
    assert row.decode_ok
    assert row.pixel_std == pytest.approx(0.0)


def test_exif_orientation_is_read(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[274] = 6
    save(tmp_path, "train/real/a.jpg", noise_image(), exif=exif)
    assert scan(tmp_path, "train/real/a.jpg").exif_orientation == 6


def test_truncated_file_is_a_failed_row(tmp_path: Path) -> None:
    path = save(tmp_path, "train/fake/a.jpg", noise_image((256, 256)), quality=95)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    row = scan(tmp_path, "train/fake/a.jpg")
    assert not row.decode_ok
    assert row.error != ""
    # header was readable before the decode failed, so these are still recorded
    assert row.width == 256
    assert row.md5 is not None


def test_not_an_image_is_a_failed_row(tmp_path: Path) -> None:
    path = tmp_path / "train/fake/a.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is not an image")
    row = scan(tmp_path, "train/fake/a.png")
    assert not row.decode_ok
    assert "UnidentifiedImageError" in row.error
    assert row.format is None and row.width is None


def test_missing_file_is_a_failed_row(tmp_path: Path) -> None:
    row = scan(tmp_path, "train/fake/ghost.png")
    assert not row.decode_ok
    assert row.file_bytes is None and row.md5 is None
    assert "FileNotFoundError" in row.error


def test_list_files_filters_folders_and_extensions(tmp_path: Path) -> None:
    save(tmp_path, "train/real/a.png", noise_image())
    save(tmp_path, "test/fake/b.JPG", noise_image())
    save(tmp_path, "processed/train/real/c.png", noise_image())  # not a split folder
    (tmp_path / "train/real/notes.txt").write_text("x")
    assert list_files(tmp_path) == ["test/fake/b.JPG", "train/real/a.png"]


@pytest.fixture
def small_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for i in range(3):
        save(root, f"train/real/r{i}.png", noise_image())
        save(root, f"train/fake/f{i}.png", noise_image())
    save(root, "test/real/r.png", noise_image())
    save(root, "test/fake/f.png", noise_image())
    return root


def test_scan_dataset_writes_csv_with_all_columns(small_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "out" / "scan.csv"
    df = scan_dataset(small_dataset, out, workers=1)
    assert len(df) == 8
    assert list(df.columns) == list(ScanRow.model_fields)
    assert out.exists()
    assert df["path"].is_monotonic_increasing


def test_scan_dataset_limit_writes_nothing(small_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "scan.csv"
    df = scan_dataset(small_dataset, out, workers=1, limit=3)
    assert len(df) == 3
    assert not out.exists()


def test_scan_dataset_limit_is_deterministic(small_dataset: Path, tmp_path: Path) -> None:
    a = scan_dataset(small_dataset, tmp_path / "x.csv", workers=1, limit=4, seed=7)
    b = scan_dataset(small_dataset, tmp_path / "x.csv", workers=1, limit=4, seed=7)
    assert list(a["path"]) == list(b["path"])


def test_scan_dataset_cache_and_force(small_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "scan.csv"
    scan_dataset(small_dataset, out, workers=1)
    (small_dataset / "test/real/r.png").unlink()
    assert len(scan_dataset(small_dataset, out, workers=1)) == 8  # served from cache
    assert len(scan_dataset(small_dataset, out, workers=1, force=True)) == 7
