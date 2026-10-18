import pandas as pd

def assign_val_split(manifest: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    
    """Return a copy of the manifest where a stratified slice of train is relabelled val."""
    
    df = manifest.sort_values("path").reset_index(drop=True)
    train = df[df["split"] == "train"]
    val_rows = train.groupby("label").sample(frac=val_fraction, random_state=seed)
    df.loc[val_rows.index, "split"] = "val"
    return df
