import pandas as pd

def window_features_classification(seg, axes=["X", "Y", "Z"]) -> pd.Series:
    f = {}
    for col in axes:
        v = seg[col].values
        f[f"{col}_mean"] = v.mean()
        f[f"{col}_std"]  = v.std()
        f[f"{col}_min"]  = v.min()
        f[f"{col}_max"]  = v.max()
    f["activity"] = seg["activity"].mode().iat[0]
    f["userid"]   = seg["userid"].iat[0]
    return pd.Series(f)

def proc_partition_classification(pdf: pd.DataFrame, win_size=512, win_step=256) -> pd.DataFrame:
    rows, start, n = [], 0, len(pdf)
    while start + win_size <= n:
        seg = pdf.iloc[start:start + win_size]
        if seg["activity"].notna().sum() == 0:
            start += win_step
            continue
        rows.append(window_features_classification(seg))
        start += win_step
    return pd.DataFrame(rows)

def extract_features_dd_classification(ddf, win_size, win_step, axes=["X", "Y", "Z"]):
    meta_cols = [f"{c}_{s}" for c in axes for s in ["mean","std","min","max"]] + ["activity", "userid"]
    meta = pd.DataFrame({c: pd.Series(dtype="float32") for c in meta_cols})
    meta["activity"] = meta["activity"].astype("category")
    meta["userid"]   = meta["userid"].astype("category")

    return ddf.map_partitions(lambda pdf: proc_partition_classification(pdf, win_size, win_step), meta=meta)


def proc_partition_raw(pdf: pd.DataFrame, win_size: int, win_step: int, axes: list = ["X", "Y", "Z"]) -> pd.DataFrame:
    """
    Processes a raw Dask DataFrame partition into windows of shape (win_size, len(axes)).

    Returns a DataFrame with columns:
        - signal: ndarray of shape (win_size, len(axes)), dtype float32
        - activity: category
        - userid: category
    """
    rows, start, n = [], 0, len(pdf)
    while start + win_size <= n:
        seg = pdf.iloc[start:start + win_size]
        if seg["activity"].notna().sum() == 0:
            start += win_step
            continue
        signal = seg[axes].astype("float32").values  # Shape: (win_size, len(axes))
        rows.append({
            "signal": signal,
            "activity": seg["activity"].mode().iat[0],
            "userid": seg["userid"].iat[0]
        })
        start += win_step
    return pd.DataFrame(rows)


def extract_features_dd_timeseries(ddf, win_size, win_step, axes=["X", "Y", "Z"]):
    # Define metadata for Dask (object dtype for ndarray)
    meta = pd.DataFrame({
        "signal": pd.Series(dtype="object"),
        "activity": pd.Series(dtype="category"),
        "userid": pd.Series(dtype="category")
    })

    return ddf.map_partitions(proc_partition_raw, win_size, win_step, axes, meta=meta)

