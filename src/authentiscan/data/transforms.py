import torch
from torchvision.transforms import v2

from authentiscan.data.stats import NormStats


def build_transform(train: bool, crop_size: int, stats: NormStats) -> v2.Compose:
    """Crop, convert to a float tensor in 0..1, then normalize. Train crops randomly, eval is fixed."""
    crop = v2.RandomCrop(crop_size) if train else v2.CenterCrop(crop_size)
    return v2.Compose([
        crop,
        v2.ToImage(),  # PIL image to a uint8 tensor (3, H, W)
        v2.ToDtype(torch.float32, scale=True),  # float32 and divide by 255
        v2.Normalize(mean=stats.mean, std=stats.std),  # (x - mean) / std per channel
    ])


def denormalize(x: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """Undo Normalize so a tensor can be viewed as an image. Works on (3, H, W) and (B, 3, H, W)."""
    mean = torch.tensor(stats.mean).view(3, 1, 1)  # (3, 1, 1) lines up with the last three dims
    std = torch.tensor(stats.std).view(3, 1, 1)
    return x * std + mean
