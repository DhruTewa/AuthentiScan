import numpy as np
import pytest
import torch
from PIL import Image

from authentiscan.data.stats import NormStats
from authentiscan.data.transforms import build_transform, denormalize

STATS = NormStats(mean=[0.5, 0.4, 0.3], std=[0.2, 0.25, 0.1], n_images=1)


def noise_image(size: int = 256) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))


@pytest.mark.parametrize("train", [True, False])
def test_output_is_a_float_tensor_of_the_crop_size(train: bool) -> None:
    out = build_transform(train, 224, STATS)(noise_image())
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (3, 224, 224)


def test_normalization_matches_hand_calculation() -> None:
    # a flat image of value v gives (v / 255 - mean) / std in every pixel of that channel
    img = Image.new("RGB", (256, 256), (255, 102, 0))
    out = build_transform(False, 224, STATS)(img)
    expected = [(1.0 - 0.5) / 0.2, (0.4 - 0.4) / 0.25, (0.0 - 0.3) / 0.1]
    for channel, value in enumerate(expected):
        assert torch.allclose(out[channel], torch.full((224, 224), value), atol=1e-5)


def test_eval_crop_is_deterministic_and_central() -> None:
    img = noise_image()
    t = build_transform(False, 224, STATS)
    assert torch.equal(t(img), t(img))
    # center crop of 256 to 224 starts at pixel 16
    raw = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
    mean = torch.tensor(STATS.mean).view(3, 1, 1)
    std = torch.tensor(STATS.std).view(3, 1, 1)
    expected = (raw[:, 16:240, 16:240] - mean) / std
    assert torch.allclose(t(img), expected, atol=1e-5)


def test_train_crop_is_random() -> None:
    img = noise_image()
    t = build_transform(True, 224, STATS)
    results = [t(img) for _ in range(8)]
    assert any(not torch.equal(results[0], r) for r in results[1:])


def test_train_crop_stays_inside_the_image() -> None:
    # every cropped value must exist in the source, so nothing was padded
    img = noise_image()
    raw = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
    mean = torch.tensor(STATS.mean).view(3, 1, 1)
    std = torch.tensor(STATS.std).view(3, 1, 1)
    full = (raw - mean) / std
    out = build_transform(True, 224, STATS)(img)
    assert out.min() >= full.min() - 1e-5
    assert out.max() <= full.max() + 1e-5


def test_same_crop_size_as_image_returns_the_whole_image() -> None:
    out = build_transform(True, 256, STATS)(noise_image(256))
    assert out.shape == (3, 256, 256)


def test_crop_larger_than_image_gets_padded_so_config_must_prevent_it() -> None:
    # documents why PreprocessConfig forbids crop_size above image_size:
    # torchvision pads or errors instead of upscaling, which would add fake pixels
    with pytest.raises(ValueError):
        build_transform(True, 300, STATS)(noise_image(256))


def test_denormalize_undoes_normalization() -> None:
    img = noise_image()
    out = build_transform(False, 224, STATS)(img)
    raw = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
    assert torch.allclose(denormalize(out, STATS), raw[:, 16:240, 16:240], atol=1e-5)


def test_denormalize_works_on_a_batch() -> None:
    batch = torch.zeros(5, 3, 8, 8)
    out = denormalize(batch, STATS)
    assert out.shape == (5, 3, 8, 8)
    assert torch.allclose(out[2, :, 0, 0], torch.tensor(STATS.mean))
