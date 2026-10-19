# AuthentiScan

End-to-end PyTorch pipeline that classifies images as AI-generated or real, with leak-aware data validation, sliced evaluation, and a FastAPI and Docker serving layer.

The goal is a binary classifier built the way a real team would build it: data pipeline, training, evaluation, serving, and monitoring. Status: work in progress. The data pipeline is built and tested (scan, filters, train/val split, preprocessing, normalization stats, `Dataset`). Model training, evaluation, serving and monitoring are next.

## Dataset

60,000 high-resolution images (about 52GB raw). Labels come from the folder names `fake` and `real`. The classes are balanced.

| Split | fake | real |
|---|---|---|
| train | 24,000 | 24,000 |
| test | 6,000 | 6,000 |

## EDA findings

The full scan and analysis are in [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).

### Shortcuts the model could learn

The two classes differ in ways unrelated to whether an image is AI-generated. A model could score well by learning these instead.

| Leak | Evidence |
|---|---|
| File format | Fakes: 10,690 PNG (36%) and 19,310 JPEG. Reals: 29,968 JPEG, 30 PNG (named `.jpg`), 1 MPO, 1 GIF. |
| Resolution and shape | 47% of sampled fakes are 1024x1024, with a median aspect ratio of 1.0. Reals vary widely, with a median aspect ratio of 1.5. |
| JPEG quality | Quantization level 4: 699 real vs 2 fake. Level 5: 343 fake vs 24 real. A rule on this value alone reaches about 75% on JPEGs. |
| Palette mode (P) | 9,051 fakes (30%) vs 3 reals. |
| Minor modes | Grayscale: 41 real vs 1 fake. RGBA: 162 fake vs 1 real. CMYK: 4 real. |
| EXIF metadata | Present on 1,531 reals and 0 fakes. 4 reals need rotation. |
| Filenames | 44,775 of 60,000 names repeat, and 15,573 names appear in both classes, so a filename is not a usable ID. |

### Data health

- **Corrupt or truncated files:** 12 (0.02%).
- **Exact duplicates (MD5):** 41 groups. 12 span train and test (leakage), 29 sit inside one split, and none span real and fake.
- **Short side under 256 px:** 13 fake, 28 real. The minimum overall is 224.
- **Aspect ratio above 3:** 112 fake, 31 real.
- **Low contrast (pixel std under 2):** 4 images.
- **Extension vs content:** 31 `.jpg` files are really PNG, plus 1 GIF and 1 MPO.

Together these affect well under 1% of the data. The shortcuts above mean preprocessing has to treat every image identically, and results will be reported sliced by format and mode, not only as one overall number.

## Data pipeline

Every step is a plain function with tests, driven by `configs/data.yaml`. Setup and commands are in [docs/dev-setup.md](docs/dev-setup.md), and the design decisions are in [docs/decisions/](docs/decisions/).

| Step | Result |
|---|---|
| Scan and filter | 59,760 images kept, 240 excluded with a reason each (broken, too small, extreme aspect ratio, blank, exact duplicate). |
| Split | Train 43,033, validation 4,781 (10%, stratified, seeded), test 11,946, untouched. |
| Preprocess | Every image resized to 256x256 and saved as PNG, so file format and size cannot reveal the class. |
| Normalize | Per-channel mean and standard deviation computed from the train split only. |
| Load | Random 224 crop for training, center 224 crop for evaluation and serving, from one shared transform. |

Re-running the scan, filter and split steps gives byte-identical CSV files (checked with md5), and re-running preprocessing skips images that already exist.

## Known limitations

- **Content and style shortcut.** In a sample batch, the fakes are mostly dramatic characters and illustrations, while the reals are mostly landscapes, objects and paintings. A high score would not by itself prove that the model detects generation artifacts. Evaluation will include a metadata-only baseline and metrics sliced by format and mode.
- **Older generators.** The fakes come from Stable Diffusion, Midjourney and DALL-E. Results say nothing about the newest generators.
- **Wide images.** Images with an aspect ratio above 3.0 are excluded, and the rest are center-cropped, so the model never sees the edges of very wide images.
