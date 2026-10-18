import logging
from pathlib import Path
from authentiscan.data.filters import apply_filters
from authentiscan.data.scan import scan_dataset
from authentiscan.schemas import load_data_config
from authentiscan.data.split import assign_val_split

CONFIG_PATH = Path("configs/data.yaml")

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_data_config(CONFIG_PATH)
    scan_df = scan_dataset(cfg.paths.raw_dir, cfg.paths.scan_path)
    kept, rejected = apply_filters(scan_df,cfg.filters)
    for df,path in (kept,cfg.paths.manifest_path), (rejected, cfg.paths.rejected_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    split_df = assign_val_split(kept, cfg.split.val_fraction, cfg.split.seed)
    cfg.paths.split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(cfg.paths.split_manifest_path, index=False)
    logging.info("Kept %d images, rejected %d images", len(kept), len(rejected))
    logging.info("Split sizes: %s", split_df["split"].value_counts().to_dict())
    
if __name__ == "__main__":
    main()