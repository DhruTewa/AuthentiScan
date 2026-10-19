import io
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from authentiscan.data.preprocess import preprocess_dataset, process_image, resize_and_crop
from authentiscan.schemas import ProcessedRow

SIZE = 256
RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


def solid(size: tuple[int, int], mode: str = "RGB", colour: object = GREEN) -> Image.Image:
    return Image.new(mode, size, colour)


@pytest.mark.parametrize(
    "size",
    [(1000, 500), (500, 1000), (512, 512), (256, 256), (300, 900), (333, 1000), (257, 800)],
)
def test_output_is_always_a_square_rgb(size: tuple[int, int]) -> None:
    out = resize_and_crop(solid(size), SIZE, "bicubic")
    assert out.size == (SIZE, SIZE)
    assert out.mode == "RGB"


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
def test_both_resize_methods_work(method: str) -> None:
    assert resize_and_crop(solid((600, 400)), SIZE, method).size == (SIZE, SIZE)


def test_unknown_method_fails() -> None:
    with pytest.raises(KeyError):
        resize_and_crop(solid((600, 400)), SIZE, "nearest")


@pytest.mark.parametrize(
    "mode, colour", [("P", 3), ("L", 128), ("RGBA", (10, 20, 30, 128)), ("1", 1)]
)
def test_other_modes_become_rgb(mode: str, colour: object) -> None:
    out = resize_and_crop(solid((400, 300), mode, colour), SIZE, "bicubic")
    assert out.mode == "RGB"


def test_crop_keeps_the_center() -> None:
    # wide image: red, green, blue thirds, the middle square must be all green
    img = Image.new("RGB", (768, 256))
    img.paste(RED, (0, 0, 256, 256))
    img.paste(GREEN, (256, 0, 512, 256))
    img.paste(BLUE, (512, 0, 768, 256))
    out = resize_and_crop(img, SIZE, "bicubic")
    assert out.getpixel((0, 128)) == GREEN
    assert out.getpixel((255, 128)) == GREEN


def test_input_image_is_not_modified() -> None:
    img = solid((600, 400))
    resize_and_crop(img, SIZE, "bicubic")
    assert img.size == (600, 400)


def test_exif_rotation_is_applied() -> None:
    # 800x200, left 40% red. EXIF 6 means "rotate clockwise to view", so red ends up on top.
    img = Image.new("RGB", (800, 200), GREEN)
    img.paste(RED, (0, 0, 320, 200))
    exif = Image.Exif()
    exif[274] = 6
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=100, exif=exif)
    buf.seek(0)
    with Image.open(buf) as tagged:
        out = resize_and_crop(tagged, SIZE, "bicubic")
    r, g, _ = out.getpixel((128, 2))  # top edge, middle column
    assert r > 200 and g < 60
    r, g, _ = out.getpixel((2, 128))  # left edge, middle row
    assert g > 200 and r < 60


def make_job(tmp_path: Path, rel: str = "train/real/0001.jpg", split: str = "train",
             label: str = "real", write: bool = True) -> tuple[str, str, str, str, str, int, str]:
    raw, out = tmp_path / "raw", tmp_path / "processed"
    if write:
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (600, 400), RED).save(raw / rel)
    return (str(raw), str(out), rel, split, label, SIZE, "bicubic")


def test_process_image_writes_a_square_png(tmp_path: Path) -> None:
    row = process_image(make_job(tmp_path))
    assert row.status == "ok" and row.error == ""
    assert row.output_path == "train/real/0001.png"
    with Image.open(tmp_path / "processed" / row.output_path) as img:
        assert img.format == "PNG"
        assert img.size == (SIZE, SIZE)


def test_output_folder_follows_the_manifest_split(tmp_path: Path) -> None:
    # a val image still lives in raw train/, but must land in processed/val/
    row = process_image(make_job(tmp_path, split="val"))
    assert row.output_path == "val/real/0001.png"
    assert (tmp_path / "processed/val/real/0001.png").exists()


def test_no_temp_file_is_left_behind(tmp_path: Path) -> None:
    process_image(make_job(tmp_path))
    assert list((tmp_path / "processed").rglob("*.tmp")) == []


def test_existing_output_is_skipped_and_not_overwritten(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    process_image(job)
    out = tmp_path / "processed/train/real/0001.png"
    before = out.stat().st_mtime_ns
    row = process_image(job)
    assert row.status == "skipped"
    assert out.stat().st_mtime_ns == before


def test_corrupt_image_is_a_failed_row_with_no_output(tmp_path: Path) -> None:
    job = make_job(tmp_path, write=False)
    bad = Path(job[0]) / job[2]
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not an image")
    row = process_image(job)
    assert row.status == "failed"
    assert "UnidentifiedImageError" in row.error
    assert not (tmp_path / "processed/train/real/0001.png").exists()


def test_missing_file_is_a_failed_row(tmp_path: Path) -> None:
    row = process_image(make_job(tmp_path, write=False))
    assert row.status == "failed"
    assert "FileNotFoundError" in row.error


def test_failed_image_can_be_retried_after_fixing(tmp_path: Path) -> None:
    job = make_job(tmp_path, write=False)
    assert process_image(job).status == "failed"
    make_job(tmp_path)  # now the file exists
    assert process_image(job).status == "ok"


@pytest.fixture
def small_manifest(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for i, (split, label) in enumerate([("train", "real"), ("train", "fake"), ("val", "real"), ("test", "fake")] * 3):
        raw_dir = "train" if split == "val" else split  # val images sit in raw train/
        rel = f"{raw_dir}/{label}/{i:03d}.jpg"
        (tmp_path / "raw" / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (400, 300), GREEN).save(tmp_path / "raw" / rel)
        rows.append({"path": rel, "split": split, "label": label})
    return pd.DataFrame(rows)


def run_dataset(manifest: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    return preprocess_dataset(manifest, tmp_path / "raw", tmp_path / "processed", SIZE, "bicubic", workers=2)


def test_dataset_first_run_is_all_ok_then_all_skipped(small_manifest: pd.DataFrame, tmp_path: Path) -> None:
    first = run_dataset(small_manifest, tmp_path)
    assert len(first) == 12
    assert (first["status"] == "ok").all()
    second = run_dataset(small_manifest, tmp_path)
    assert (second["status"] == "skipped").all()
    assert len(list((tmp_path / "processed").rglob("*.png"))) == 12


def test_dataset_columns_and_order_follow_the_manifest(small_manifest: pd.DataFrame, tmp_path: Path) -> None:
    out = run_dataset(small_manifest, tmp_path)
    assert list(out.columns) == list(ProcessedRow.model_fields)
    assert list(out["path"]) == list(small_manifest["path"])


def test_dataset_one_bad_image_does_not_stop_the_rest(small_manifest: pd.DataFrame, tmp_path: Path) -> None:
    (tmp_path / "raw" / small_manifest.loc[3, "path"]).write_bytes(b"broken")
    out = run_dataset(small_manifest, tmp_path)
    assert out["status"].value_counts().to_dict() == {"ok": 11, "failed": 1}
    assert out.loc[3, "status"] == "failed"


def test_dataset_empty_manifest_gives_empty_table_with_header(tmp_path: Path) -> None:
    empty = pd.DataFrame({"path": [], "split": [], "label": []})
    out = run_dataset(empty, tmp_path)
    assert out.empty
    assert list(out.columns) == list(ProcessedRow.model_fields)
