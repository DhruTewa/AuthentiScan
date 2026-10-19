import logging
import sys
from pathlib import Path

import pandas as pd

from authentiscan.data.preprocess import preprocess_dataset
from authentiscan.schemas import load_data_config

CONFIG_PATH = Path("configs/data.yaml")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_data_config(CONFIG_PATH)
    manifest = pd.read_csv(cfg.paths.split_manifest_path)
    #manifest = pd.read_csv(cfg.paths.split_manifest_path).sample(200, random_state=0)

  
    results = preprocess_dataset(manifest, cfg.paths.raw_dir, cfg.paths.processed_dir,
                cfg.preprocess.image_size, cfg.preprocess.resize_method)
    usable = results[results["status"].isin(["ok", "skipped"])]
    failed = results[results["status"] == "failed"]
    cfg.paths.processed_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    usable.to_csv(cfg.paths.processed_manifest_path, index=False)
    logging.info("Processed %d, failed %d", len(usable), len(failed))
    if len(failed):
        logging.error("Failed images:\n%s", failed[["path", "error"]].to_string(index=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
