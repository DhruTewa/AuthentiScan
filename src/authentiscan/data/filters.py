import pandas as pd

from authentiscan.schemas import FiltersConfig


def apply_filters(
    scan: pd.DataFrame, cfg: FiltersConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the scan table into (kept, rejects). Rejects get a `reject_reason` column."""
    scan = scan.sort_values("path").reset_index(drop=True)  # same order every run

    # one boolean mask per rule, True means "reject this row"
    short_side = scan[["width", "height"]].min(axis=1)
    masks: dict[str, pd.Series] = {
        "decode_failed": ~scan["decode_ok"],
        "too_small": short_side < cfg.min_short_side,
        "extreme_aspect": scan["aspect_ratio"] > cfg.max_aspect_ratio,
        "low_contrast": scan["pixel_std"] < cfg.min_pixel_std,
    }

    # failed decodes have NaN measurements, so their comparisons are False and
    # only the decode rule flags them
    reasons = pd.DataFrame(masks)
    rejected = reasons.any(axis=1)
    scan["reject_reason"] = [";".join(reasons.columns[row]) for row in reasons.to_numpy()]

    if cfg.dedupe_exact:
        # only rows that passed every other rule compete to be the surviving copy
        alive = scan[~rejected].copy()
        alive["rank"] = (alive["split"] == "test").astype(int)  # train copy sorts first
        alive = alive.sort_values(["md5", "rank", "path"])
        dup_index = alive.index[alive.duplicated(subset="md5", keep="first")]
        scan.loc[dup_index, "reject_reason"] = "duplicate_exact"

    is_rejected = scan["reject_reason"] != ""
    return scan[~is_rejected], scan[is_rejected]
