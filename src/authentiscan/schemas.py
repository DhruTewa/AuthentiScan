from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PathsConfig(BaseModel):
    # forbid unknown keys so a typo in the YAML fails loudly
    model_config = ConfigDict(extra="forbid")
    raw_dir: Path
    processed_dir: Path
    manifest_path: Path


class ClassesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fake: int
    real: int

    @model_validator(mode="after")
    def labels_differ(self) -> "ClassesConfig":
        # same label for both classes would make training learn nothing
        if self.fake == self.real:
            raise ValueError("fake and real need different labels")
        return self


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    val_fraction: float = Field(gt=0, lt=1)


class PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_size: int = Field(gt=0)
    resize_method: Literal["bicubic", "lanczos"]
    output_format: Literal["PNG"]  # lossless, so no new compression artifacts
    mode: Literal["RGB"]


class FiltersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_short_side: int = Field(gt=0)
    max_aspect_ratio: float = Field(gt=1)
    min_pixel_std: float = Field(ge=0)
    dedupe_exact: bool


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paths: PathsConfig
    classes: ClassesConfig
    split: SplitConfig
    preprocess: PreprocessConfig
    filters: FiltersConfig

    @model_validator(mode="after")
    def no_upscaling(self) -> "DataConfig":
        # upscaled images carry a blur that would differ by class
        if self.filters.min_short_side < self.preprocess.image_size:
            raise ValueError("min_short_side must be at least image_size")
        return self


def load_data_config(path: Path) -> DataConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return DataConfig.model_validate(data)
