import logging
from pathlib import Path
import pandas as pd
from authentiscan.data.stats import compute_norm_stats
from authentiscan.schemas import load_data_config

CONFIG_PATH = Path("configs/data.yaml")

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_data_config(CONFIG_PATH)
    manifest = pd.read_csv(cfg.paths.processed_manifest_path)
    stats = compute_norm_stats(manifest, cfg.paths.processed_dir)
    cfg.paths.norm_stats_path.write_text(stats.model_dump_json(indent=2))
    logging.info("mean %s std %s from %d train images", stats.mean, stats.std, stats.n_images)
    
if __name__ == "__main__":
    main()
    