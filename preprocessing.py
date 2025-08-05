from pathlib import Path
import dask.dataframe as dd

BASE_COLS = ["Creation_Time", "x", "y", "z", "gt", "User"]
DTYPES = {
    "Creation_Time": "int64",
    "x": "float32",
    "y": "float32",
    "z": "float32",
    "gt": "category",
    "User": "category",
}

def get_file_paths(base_path: Path):
    return {
        "acc_phone":  base_path / "Activity recognition exp" / "Phones_accelerometer.csv",
        "gyro_phone": base_path / "Activity recognition exp" / "Phones_gyroscope.csv",
        "acc_watch":  base_path / "Activity recognition exp" / "Watch_accelerometer.csv",
        "gyro_watch": base_path / "Activity recognition exp" / "Watch_gyroscope.csv",
    }

def load_sensor(path: Path, tag: str) -> dd.DataFrame:
    df = dd.read_csv(
        path,
        usecols=BASE_COLS,
        dtype=DTYPES,
        blocksize="128MB",
        engine="pyarrow",
        assume_missing=True,
    ).rename(columns={
        "Creation_Time": "time",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "gt": "activity",
        "User": "userid",
    })
    df = df.assign(sensor_tag=tag)
    return df

def preprocess_dataset(
    base_path: str,
    sensor_key: str = "acc_phone",
    partition_size: str = "256M",
    downsample_stride: int = 1
):
    base_path = Path(base_path)
    files = get_file_paths(base_path)

    path = files.get(sensor_key)
    if not path or not path.exists():
        raise ValueError(f"Sensor key '{sensor_key}' not found or file does not exist: {path}")

    df = load_sensor(path, tag=sensor_key)
    df = df.dropna(subset=["activity"])

    # Count before subsampling
    original_rows = df.shape[0].compute()

    # Set 'time' as index (sorted and partitioned)
    df = df.set_index("time", shuffle="tasks", partition_size=partition_size, sort=True)

    if downsample_stride > 1:
        # Subsampling by row position

        # Reset index for manipulation
        df = df.reset_index(drop=False)

        # Compute partition lengths and offsets
        partition_lengths = df.map_partitions(len).compute()
        partition_offsets = partition_lengths.cumsum() - partition_lengths

        col_order = df.columns.tolist()

        def add_row_id(partition, offset, col_order):
            partition = partition.copy()
            partition['row_id'] = pd.RangeIndex(start=offset, stop=offset + len(partition))
            return partition[col_order + ['row_id']]

        # Apply to all partitions
        df = dd.concat([
            df.partitions[i].map_partitions(add_row_id, offset, col_order)
            for i, offset in enumerate(partition_offsets)
        ])

        # Subsample every n-th row
        df = df[df['row_id'] % downsample_stride == 0].drop(columns='row_id')

        # Set 'time' back as index
        df = df.set_index('time', shuffle="tasks", sort=True)

    # Count after subsampling
    final_rows = df.shape[0].compute()

    # Report summary
    percent = (final_rows / original_rows) * 100
    print(f"Subsampling applied: keeping 1 every {downsample_stride} rows")
    print(f"Rows before: {original_rows:,}")
    print(f"Rows after:  {final_rows:,}")
    print(f"Retained:    {percent:.2f}%")

    return df.persist()