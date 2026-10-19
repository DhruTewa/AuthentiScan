# 0002: Reject images with an aspect ratio above 3.0

Status: Accepted (2026-10-18)

## Context
Aspect ratio (long side divided by short side) differs by class. In the scan, fakes have a median of 1.0 and real images a median of 1.5. A few images are extremely wide or tall, with a maximum of 16.28 among fakes and 5.91 among reals.

Training uses a fixed square input (resize to 256, crop to 224). An image with a ratio of 5 or 16 either gets squashed, which distorts the content, or cropped, which throws most of it away. The model would then learn from a very different picture than the label describes.

## Decision
Reject images with `aspect_ratio > 3.0`. The threshold is `filters.max_aspect_ratio` in `configs/data.yaml`.

## Effect on the data
143 images exceed 3.0: 112 fake and 31 real. One of the real ones is also too small, so 142 rows carry `extreme_aspect` as their only reason and 1 carries `too_small;extreme_aspect`. This is the largest single cause of rejection, and it is why rejects skew toward fakes (163 fake vs 77 real in total, or 0.54% of fakes vs 0.26% of reals).

## Alternatives considered
- **Keep them and squash to a square.** Rejected, because distortion adds an artefact that only some images have.
- **Pad to a square.** Rejected, because the padding itself would become a feature, and the ratio differs by class.
- **A higher limit such as 5.** Not tested. It would keep a few more images but does not remove the distortion problem.

## Consequences
- The model never sees panoramas or banners. It should not be trusted on them.
- Serving should apply the same rule: warn or refuse for uploads above ratio 3.0, instead of silently squashing them.
- The loss is small (0.37% of fakes, 0.10% of reals), but it is uneven. This is recorded so a reader does not mistake it for an accident.
