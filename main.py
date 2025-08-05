# main.py
import yaml
from pathlib import Path
from dask.distributed import Client
from preprocessing import preprocess_dataset
from processing import extract_features_dd_classification, extract_features_dd_timeseries
from classification import train_model
from timeSeries import train_time_series_model, train_kmeans_timeseries
from customLogger import Tee
import sys
import time

classification_models = ['xgboost', 'random_forest', 'gbc']
time_series_models = ['cnn', 'tcn', 'tsc']

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <config_file.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.suffix.lower() in [".yaml", ".yml"]:
        print("Error: Config file must have a .yaml or .yml extension.")
        sys.exit(1)

    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)

    start_time = time.time()
    config = load_config(config_path)
    output_name = config.get("output", {}).get("file_name", "output_log")
    output_path = Path(f"{output_name}.txt")
    with output_path.open("w") as logfile, Client() as client:
        # Redirect both stdout and stderr to the same Tee object
        tee = Tee(sys.__stdout__, logfile)
        sys.stdout = tee
        sys.stderr = tee

        print("Dask client initialized")

        # Preprocess data
        dataset_cfg = config["dataset"]
        ACC = preprocess_dataset(
            base_path=dataset_cfg["base_path"],
            sensor_key=dataset_cfg["sensor"],
            partition_size=dataset_cfg["partition_size"],
            downsample_stride=dataset_cfg["downsample_stride"]
        )
        print(f"Dataset '{dataset_cfg['sensor']}' loaded with {ACC.npartitions} partitions")
        training_cfg = config["training"]
        features_cfg = config["features"]
        if training_cfg["model"] in classification_models:
            print(f"Training classification model: {training_cfg['model']}")
            features_dd = extract_features_dd_classification(ACC, win_size=features_cfg["window_size"], win_step=features_cfg["step_size"])
            features = features_dd.compute()
            print(f"Features extracted for '{dataset_cfg['sensor']}' with {features.shape[0]} samples")
            train_model(features, training_cfg["model"])
        if training_cfg["model"] in time_series_models:
            print(f"Training time-series model: {training_cfg['model']}")
            windows_dd = extract_features_dd_timeseries(
                ACC,
                win_size=features_cfg["window_size"],
                win_step=features_cfg["step_size"]
            )
            windows = windows_dd.compute()
            print(f"✅ Time-series windows extracted: {windows.shape[0]}")
            if training_cfg["model"] == "tsc":
                train_kmeans_timeseries(windows, n_clusters=6, method="sktime")
            else:
                train_time_series_model(windows, training_cfg["model"])
        
        total_time = (time.time() - start_time)/60
        print(f"Total processing time: {total_time:.2f} minutes")

if __name__ == "__main__":
    main()